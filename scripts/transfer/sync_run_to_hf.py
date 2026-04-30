#!/usr/bin/env python3
"""Sync a completed training run to Hugging Face Hub, then optionally clean up.

Designed to run on the **VPS** (which can reach both Ensimag via SSH and
HF Hub over HTTPS). Usage from the VPS:

    python sync_run_to_hf.py \\
        --remote-host ensimag \\
        --remote-output-dir /user/2/zimmermh/outputs/phi2_harder_multitask_seed42 \\
        --hf-repo Helain/gated-lora-experiments \\
        --hf-path phi2_harder_multitask_seed42 \\
        [--keep-latest]   # don't pull/delete the LATEST checkpoint (still resumable)
        [--cleanup]        # delete pulled files from remote after successful upload

It can also run locally (no SSH) when --remote-host is omitted — just point
--remote-output-dir at a local directory.

Behavior:
  1. Build a list of files under remote-output-dir (skip optimizer.pt by
     default since they rotate; pass --include-optimizer to override).
  2. Skip files in checkpoint-LATEST/ if --keep-latest (so the SLURM chain
     can still resume from it).
  3. rsync everything matching to a temp staging dir on the VPS.
  4. huggingface_hub.upload_folder() to the dataset repo at <hf-path>/.
  5. If --cleanup, delete the synced files from the remote.

The marker file `TRAINING_DONE` is **never deleted** — it's the signal that
the chain orchestrator uses to stop resubmitting.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote-host", default=None,
                   help="SSH host alias (e.g. 'ensimag'). If omitted, --remote-output-dir is treated as local.")
    p.add_argument("--remote-output-dir", required=True,
                   help="Path on remote (or local) holding outputs/<run>")
    p.add_argument("--hf-repo", required=True, help="HF dataset repo (e.g. Helain/gated-lora-experiments)")
    p.add_argument("--hf-path", required=True, help="Subdirectory within the HF repo")
    p.add_argument("--keep-latest", action="store_true",
                   help="Skip checkpoint-LATEST/* (still needed for SLURM resume)")
    p.add_argument("--include-optimizer", action="store_true",
                   help="Include optimizer.pt files (default: skip — they rotate)")
    p.add_argument("--cleanup", action="store_true",
                   help="Delete synced files from remote after upload succeeds")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without performing them")
    return p.parse_args()


def run(cmd: list[str] | str, *, dry_run: bool = False) -> str:
    """Run a shell command, raising on failure. Returns stdout."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    logger.info(f"$ {cmd_str}")
    if dry_run:
        return ""
    result = subprocess.run(
        cmd if isinstance(cmd, list) else cmd_str,
        shell=isinstance(cmd, str),
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        logger.debug(result.stdout)
    return result.stdout


def build_rsync_excludes(args: argparse.Namespace) -> list[str]:
    excludes = ["--exclude=*.pyc", "--exclude=__pycache__"]
    if args.keep_latest:
        excludes.append("--exclude=checkpoint-LATEST/")
        excludes.append("--exclude=latest/")
    if not args.include_optimizer:
        excludes.append("--exclude=optimizer.pt")
    return excludes


def build_remote_path(args: argparse.Namespace) -> str:
    """Format the source path for rsync — adds 'host:' prefix if remote."""
    if args.remote_host:
        return f"{args.remote_host}:{args.remote_output_dir}/"
    return f"{args.remote_output_dir}/"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN env var must be set")
        return 1

    with tempfile.TemporaryDirectory(prefix="gated-lora-sync-") as staging:
        staging_path = Path(staging) / Path(args.remote_output_dir).name
        staging_path.mkdir(parents=True, exist_ok=True)

        # 1. rsync from remote to staging
        rsync_cmd = (
            ["rsync", "-av", "--info=progress2"]
            + build_rsync_excludes(args)
            + [build_remote_path(args), str(staging_path)]
        )
        run(rsync_cmd, dry_run=args.dry_run)

        if args.dry_run:
            logger.info(f"--dry-run: would upload {staging_path} → hf://{args.hf_repo}/{args.hf_path}")
            return 0

        # 2. huggingface_hub upload (deferred import — avoids hard dep on local-only runs)
        try:
            from huggingface_hub import HfApi, login
        except ImportError:
            logger.error("huggingface_hub not installed: `pip install huggingface_hub`")
            return 1

        login(token=hf_token, add_to_git_credential=False)
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging_path),
            repo_id=args.hf_repo,
            repo_type="dataset",
            path_in_repo=args.hf_path,
            commit_message=f"Sync {Path(args.remote_output_dir).name}",
        )
        logger.info(f"Uploaded → hf://{args.hf_repo}/{args.hf_path}")

        # 3. Cleanup remote (only if explicitly requested AND upload succeeded)
        if args.cleanup:
            logger.info(f"Cleaning up remote {build_remote_path(args)} ...")
            # Delete files we synced (everything except TRAINING_DONE and excluded patterns)
            cleanup_excludes = build_rsync_excludes(args)
            cleanup_excludes.append("--exclude=TRAINING_DONE")
            # Use rsync --delete with same source/excludes against an empty source
            # Simpler: SSH and find -delete
            if args.remote_host:
                find_cmd = (
                    f"find {shlex.quote(args.remote_output_dir)} -type f "
                    f"! -name TRAINING_DONE "
                    f"{'! -name optimizer.pt' if not args.include_optimizer else ''} "
                    f"{'! -path \\*/checkpoint-LATEST/\\*' if args.keep_latest else ''} "
                    f"-delete"
                )
                run(["ssh", args.remote_host, find_cmd])
            else:
                # Local cleanup
                for f in Path(args.remote_output_dir).rglob("*"):
                    if not f.is_file():
                        continue
                    if f.name == "TRAINING_DONE":
                        continue
                    if not args.include_optimizer and f.name == "optimizer.pt":
                        continue
                    if args.keep_latest and "checkpoint-LATEST" in f.parts:
                        continue
                    if args.keep_latest and "latest" in f.parts:
                        continue
                    f.unlink()

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
