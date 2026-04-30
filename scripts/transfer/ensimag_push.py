#!/usr/bin/env python3
"""Push training artifacts from Ensimag → Hugging Face Hub, then delete locally.

Designed to run **on the Ensimag login node** (since the OVH VPS cannot reach
Ensimag through the school firewall, but Ensimag itself has outbound HTTPS
to huggingface.co).

For each run directory under ``outputs/``:

1. Scan ``checkpoint-N/`` directories (not ``checkpoint-LATEST/`` / ``latest/``).
2. For each one not yet recorded in ``.uploaded`` of that run:
   - Upload its contents (excluding optimizer.pt) to
     ``hf://Helain/gated-lora-experiments/<run>/<checkpoint-N>/``.
   - Delete the checkpoint directory locally.
   - Append the checkpoint name to ``.uploaded``.
3. If ``TRAINING_DONE`` is present in the run dir:
   - Upload any remaining root-level files (``final_results.json``,
     ``routing_history.json``, ``experiment_config.json``, figures) under
     ``hf://.../<run>/``.
   - Delete everything in the run dir EXCEPT ``TRAINING_DONE`` (the marker
     stays so ``chain_jobs.sh`` knows to stop resubmitting).

Idempotent: re-running is safe (already-uploaded checkpoints are skipped).

Usage (Ensimag cron, every 5 min):
    */5 * * * * /path/to/scripts/transfer/ensimag_cron.sh >> ~/glr-push.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)

# Files we never upload (rotated by trainer, only useful for local resume)
EXCLUDE_FILES = {"optimizer.pt"}
# Subdirs we never touch (live state for SLURM resume)
KEEP_DIRS = {"checkpoint-LATEST", "latest"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", required=True,
                   help="Path to outputs/ holding <run>/ subdirectories")
    p.add_argument("--hf-repo", default="Helain/gated-lora-experiments",
                   help="HF dataset repo (default: Helain/gated-lora-experiments)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without uploading or deleting")
    p.add_argument("--verbose", action="store_true",
                   help="DEBUG-level logging")
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Quiet down chatty libraries — keep our own logger at the requested level.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def list_completed_checkpoints(run_dir: Path) -> List[Path]:
    """Return checkpoint-N/ dirs that are NOT the live LATEST.

    Trainer guarantees: at most one ``checkpoint-LATEST`` (or ``latest``)
    holds optimizer.pt. All other ``checkpoint-N/`` are safe to ship.
    """
    out = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in KEEP_DIRS:
            continue
        if child.name.startswith("checkpoint-") or child.name in {"best_model", "final_model"}:
            out.append(child)
    return out


def already_uploaded(run_dir: Path, checkpoint_name: str) -> bool:
    state = run_dir / ".uploaded"
    if not state.exists():
        return False
    return checkpoint_name in state.read_text().splitlines()


def mark_uploaded(run_dir: Path, checkpoint_name: str) -> None:
    state = run_dir / ".uploaded"
    with open(state, "a") as f:
        f.write(checkpoint_name + "\n")


def upload_dir(api, src: Path, hf_repo: str, hf_path: str) -> None:
    """Upload ``src`` to ``hf://hf_repo/hf_path/``, excluding EXCLUDE_FILES."""
    api.upload_folder(
        folder_path=str(src),
        repo_id=hf_repo,
        repo_type="dataset",
        path_in_repo=hf_path,
        ignore_patterns=[f"**/{x}" for x in EXCLUDE_FILES],
        commit_message=f"Sync {hf_path}",
    )


def upload_run_root_files(api, run_dir: Path, hf_repo: str, hf_path: str) -> List[Path]:
    """Upload root-level non-checkpoint files. Returns list of files uploaded."""
    candidates = [
        run_dir / name
        for name in (
            "final_results.json",
            "routing_history.json",
            "experiment_config.json",
        )
    ]
    candidates.extend(run_dir.glob("*.png"))
    candidates.extend(run_dir.glob("*.md"))
    existing = [c for c in candidates if c.exists() and c.is_file()]
    if not existing:
        return []
    for f in existing:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"{hf_path}/{f.name}",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Sync {hf_path}/{f.name}",
        )
    return existing


def safe_rmtree(path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info(f"[dry-run] rmtree: {path}")
        return
    shutil.rmtree(path)
    logger.info(f"  removed: {path}")


def safe_unlink(path: Path, dry_run: bool) -> None:
    if dry_run:
        logger.info(f"[dry-run] unlink: {path}")
        return
    path.unlink()


def process_run(api, run_dir: Path, hf_repo: str, dry_run: bool) -> None:
    run_name = run_dir.name
    logger.info(f"=== {run_name} ===")

    # Phase 1: ship completed checkpoints (skip LATEST, skip already-uploaded)
    for ckpt in list_completed_checkpoints(run_dir):
        if already_uploaded(run_dir, ckpt.name):
            logger.debug(f"  already uploaded: {ckpt.name}")
            continue

        logger.info(f"  uploading {ckpt.name}")
        if not dry_run:
            upload_dir(api, ckpt, hf_repo, f"{run_name}/{ckpt.name}")
            mark_uploaded(run_dir, ckpt.name)
        safe_rmtree(ckpt, dry_run)

    # Phase 2: if run is complete, ship root files + clean up everything but the marker
    done_marker = run_dir / "TRAINING_DONE"
    if done_marker.exists():
        logger.info(f"  TRAINING_DONE present — final cleanup")
        if not dry_run:
            uploaded_root_files = upload_run_root_files(api, run_dir, hf_repo, run_name)
            logger.info(f"  uploaded {len(uploaded_root_files)} root file(s)")

        # Delete everything except TRAINING_DONE (marker remains for chain_jobs.sh)
        for child in list(run_dir.iterdir()):
            if child.name == "TRAINING_DONE":
                continue
            if child.is_dir():
                safe_rmtree(child, dry_run)
            else:
                safe_unlink(child, dry_run)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run:
        logger.error("HF_TOKEN env var must be set (use --dry-run to test without it)")
        return 1

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    if not outputs_dir.exists():
        logger.info(f"No outputs/ at {outputs_dir} — nothing to do")
        return 0

    if args.dry_run:
        api = None
    else:
        from huggingface_hub import HfApi, login
        login(token=hf_token, add_to_git_credential=False)
        api = HfApi()

    run_dirs = [d for d in sorted(outputs_dir.iterdir()) if d.is_dir()]
    if not run_dirs:
        logger.info("No run directories — nothing to do")
        return 0

    for rd in run_dirs:
        try:
            process_run(api, rd, args.hf_repo, args.dry_run)
        except Exception as exc:
            logger.error(f"  FAILED on {rd.name}: {exc}")
            # Continue with other runs instead of aborting
            continue

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
