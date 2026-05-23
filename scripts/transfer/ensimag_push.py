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

# Files we never upload (transient compute-local caches)
EXCLUDE_FILES = set()  # optimizer.pt MUST go to HF for stage-in resume across jobs

# In stage-in/stage-out mode on Ensimag compute, /tmp is wiped between jobs.
# The "latest" checkpoint therefore MUST be pushed to HF so the next job can
# pull it for resume. But we ALSO keep it locally during the same job for the
# trainer's in-process resume logic.
KEEP_LOCAL_AFTER_PUSH = {"checkpoint-LATEST", "latest"}


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


def list_all_checkpoint_dirs(run_dir: Path) -> List[Path]:
    """All checkpoint-shaped subdirs (numbered + named like best_model/final_model + latest)."""
    out = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("checkpoint-") or child.name in {"best_model", "final_model", "latest"}:
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
    """Upload root-level non-checkpoint files. Returns list of files uploaded.

    TRAINING_DONE is included so login-side orchestrators (chain_jobs.sh) can
    discover completion via HF API without needing access to compute /tmp.
    """
    candidates = [
        run_dir / name
        for name in (
            "final_results.json",
            "routing_history.json",
            "experiment_config.json",
            "TRAINING_DONE",
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


def push_single_run(run_dir: Path, hf_repo: str = "Helain/gated-lora-experiments") -> None:
    """Library entry point: push a single run dir, called by the trainer post-save.

    Re-authenticates against HF Hub on every call (compute nodes lose auth state
    between SLURM jobs, so we can't assume a prior login is still valid).
    Logs and swallows exceptions: a failed push must NOT crash training.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN unset — skipping push for %s", run_dir.name)
        return
    try:
        from huggingface_hub import HfApi, login
        login(token=hf_token, add_to_git_credential=False)
        api = HfApi()
        process_run(api, run_dir, hf_repo, dry_run=False)
    except Exception as exc:
        logger.warning("Push failed for %s: %s (training continues)", run_dir.name, exc)


def process_run(api, run_dir: Path, hf_repo: str, dry_run: bool) -> None:
    run_name = run_dir.name
    logger.info(f"=== {run_name} ===")

    # Phase 1: ship every checkpoint dir.
    # - Numbered checkpoints + best_model + final_model: push once (tracked in
    #   .uploaded), then delete locally.
    # - "latest" / "checkpoint-LATEST": always re-push (content changes each save)
    #   and KEEP local — the in-process trainer needs it for resume within the
    #   current job; the next SLURM job pulls it from HF for cross-job resume.
    for ckpt in list_all_checkpoint_dirs(run_dir):
        is_latest = ckpt.name in KEEP_LOCAL_AFTER_PUSH

        if is_latest:
            logger.info(f"  uploading {ckpt.name} (live state, kept local)")
            if not dry_run:
                upload_dir(api, ckpt, hf_repo, f"{run_name}/{ckpt.name}")
            # NEVER delete latest
            continue

        if already_uploaded(run_dir, ckpt.name):
            logger.debug(f"  already uploaded: {ckpt.name}")
            # Still safe to clean up if it's somehow stuck around
            if ckpt.exists():
                safe_rmtree(ckpt, dry_run)
            continue

        logger.info(f"  uploading {ckpt.name}")
        if not dry_run:
            upload_dir(api, ckpt, hf_repo, f"{run_name}/{ckpt.name}")
            mark_uploaded(run_dir, ckpt.name)
        safe_rmtree(ckpt, dry_run)

    # Phase 2: if run is complete, ship root files (incl. TRAINING_DONE).
    # We do NOT wipe the run_dir here — /tmp is wiped by SLURM at job end.
    done_marker = run_dir / "TRAINING_DONE"
    if done_marker.exists():
        logger.info(f"  TRAINING_DONE present — uploading run metadata")
        if not dry_run:
            uploaded_root_files = upload_run_root_files(api, run_dir, hf_repo, run_name)
            logger.info(f"  uploaded {len(uploaded_root_files)} root file(s)")


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
