#!/usr/bin/env python3
"""Push a run's v2 artifacts to the Hugging Face Hub dataset.

v2 layout (contract 2026-10-07): a run dir only holds
``latest/`` (resume state), ``best_model/``, ``final_model/``, root files
(``final_results.json``, ``experiment_config.json``, figures...) and the
``TRAINING_DONE`` marker. Numbered ``checkpoint-N/`` and ``epoch-N/`` dirs no
longer exist and are NEVER pushed (legacy dirs found locally are ignored and
reported, not uploaded, not deleted).

Uploaded to ``hf://<repo>/<run>/<name>/`` where ``<run>`` is the run dir name
(= GLR_RUN_NAME in SLURM jobs). Nothing is deleted locally: /tmp WORK dirs are
managed by train.sbatch.

The v2 trainer pushes by itself. This module stays as:
  * ``push_single_run(run_dir)`` (backward compat for trainer code that still
    imports it): best effort, never raises;
  * a CLI for manual catch-up: ``--outputs-dir <dir holding <run>/ dirs>``.

Authentication: the token is taken from the HF_TOKEN env var by
huggingface_hub. We never call ``login()``, which writes the token to
``$HF_HOME/token`` (a node-shared /tmp cache on the cluster).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

PUSHED_DIRS = ("latest", "best_model", "final_model")
LEGACY_DIR_RE = re.compile(r"^(checkpoint|epoch)-\d+$|^checkpoint-LATEST$")
ROOT_FILES = ("final_results.json", "routing_history.json", "experiment_config.json")
DONE_MARKER = "TRAINING_DONE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", required=True, help="Directory holding <run>/ subdirectories")
    p.add_argument("--hf-repo", default=os.environ.get("GLR_HF_REPO", "Helain/gated-lora-experiments"))
    p.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def root_files(run_dir: Path) -> List[Path]:
    """Root-level files to ship once the run is done. TRAINING_DONE goes LAST."""
    files = [run_dir / n for n in ROOT_FILES]
    files += sorted(run_dir.glob("*.png")) + sorted(run_dir.glob("*.md"))
    return [f for f in files if f.is_file()]


def process_run(api, run_dir: Path, hf_repo: str, dry_run: bool) -> None:
    run_name = run_dir.name
    logger.info(f"=== {run_name} ===")

    legacy = [c.name for c in run_dir.iterdir() if c.is_dir() and LEGACY_DIR_RE.match(c.name)]
    if legacy:
        logger.warning(f"  ignoring legacy dirs (not pushed in v2): {sorted(legacy)}")

    for name in PUSHED_DIRS:
        src = run_dir / name
        if not (src.is_dir() and any(src.iterdir())):
            continue
        logger.info(f"  uploading {name}/")
        if not dry_run:
            api.upload_folder(
                folder_path=str(src), repo_id=hf_repo, repo_type="dataset",
                path_in_repo=f"{run_name}/{name}", commit_message=f"Sync {run_name}/{name}",
            )

    done = run_dir / DONE_MARKER
    if done.is_file():
        files = root_files(run_dir) + [done]
        logger.info(f"  {DONE_MARKER} present: uploading {len(files)} root file(s)")
        for f in files:
            if dry_run:
                logger.info(f"  [dry-run] would upload {f.name}")
                continue
            api.upload_file(
                path_or_fileobj=str(f), path_in_repo=f"{run_name}/{f.name}",
                repo_id=hf_repo, repo_type="dataset", commit_message=f"Sync {run_name}/{f.name}",
            )
        if not dry_run and not api.file_exists(hf_repo, f"{run_name}/{DONE_MARKER}", repo_type="dataset"):
            raise RuntimeError(f"{DONE_MARKER} not visible on HF after upload")


def push_single_run(run_dir: Path, hf_repo: str | None = None) -> None:
    """Best-effort push of one run dir; logs and swallows every error."""
    hf_repo = hf_repo or os.environ.get("GLR_HF_REPO", "Helain/gated-lora-experiments")
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN unset: skipping push for %s", Path(run_dir).name)
        return
    try:
        from huggingface_hub import HfApi
        process_run(HfApi(), Path(run_dir), hf_repo, dry_run=False)
    except Exception as exc:
        logger.warning("Push failed for %s: %s (training continues)", Path(run_dir).name, exc)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    if not os.environ.get("HF_TOKEN") and not args.dry_run:
        logger.error("HF_TOKEN env var must be set (use --dry-run to test without it)")
        return 1
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    run_dirs = [d for d in sorted(outputs_dir.iterdir()) if d.is_dir()] if outputs_dir.is_dir() else []
    if not run_dirs:
        logger.info(f"No run directories under {outputs_dir}: nothing to do")
        return 0
    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
    failures = 0
    for rd in run_dirs:
        try:
            process_run(api, rd, args.hf_repo, args.dry_run)
        except Exception as exc:
            failures += 1
            logger.error(f"  FAILED on {rd.name}: {exc}")
    logger.info("Done.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
