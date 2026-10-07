#!/usr/bin/env python3
"""Shrink the history of the HF dataset (default Helain/gated-lora-experiments).

Every push of latest/ or best_model/ adds a commit and keeps the old blobs
in the git history, so repo storage grows with each slice. This tool:

  * ``--whole-repo``: squash the whole branch into one commit
    (``HfApi.super_squash_history``). Old blobs become unreferenced and are
    garbage-collected by the Hub.
  * ``--run <path>``: git has no per-path history, so a squash is always
    branch-wide. For one run path this tool first deletes the run's legacy
    ``checkpoint-N/`` / ``epoch-N/`` / ``checkpoint-LATEST/`` folders (one
    commit), then squashes the branch. Use ``--no-squash`` to only prune.

DRY-RUN BY DEFAULT: prints what would happen. Pass ``--execute`` to act.
Squashing is irreversible (all previous revisions disappear) and must not
race with a running job pushing to the repo: the tool refuses when
``squeue -u $USER`` lists jobs, unless ``--ignore-running-jobs``.

Token: HF_TOKEN env var, else ``${GLR_HF_TOKEN_FILE:-~/.hf_token}``.

Examples:
  python scripts/maintenance/squash_hf_history.py --whole-repo
  python scripts/maintenance/squash_hf_history.py --run phi2_harder_multitask_seed42 --execute
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LEGACY_RE = re.compile(r"^(checkpoint-\d+|epoch-\d+|checkpoint-LATEST)$")


def load_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = Path(os.environ.get("GLR_HF_TOKEN_FILE", Path.home() / ".hf_token"))
    if not f.is_file():
        return None
    for line in f.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "HF_TOKEN=" in line:
            line = line.split("HF_TOKEN=", 1)[1]
        elif "=" in line:
            continue
        line = line.strip().strip("\"'")
        if line:
            return line
    return None


def running_jobs() -> str:
    if not shutil.which("squeue"):
        return ""
    user = os.environ.get("USER", "")
    try:
        out = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i %j %T"],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return ""
    return out.stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", help="Run path inside the repo (e.g. phi2_harder_multitask_seed42)")
    target.add_argument("--whole-repo", action="store_true", help="Squash the whole branch")
    p.add_argument("--repo", default=os.environ.get("GLR_HF_REPO", "Helain/gated-lora-experiments"))
    p.add_argument("--branch", default="main")
    p.add_argument("--no-squash", action="store_true", help="With --run: only prune legacy folders")
    p.add_argument("--execute", action="store_true", help="Actually modify the repo (default: dry run)")
    p.add_argument("--ignore-running-jobs", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi

    token = load_token()
    if not token:
        print("ERROR: no HF token (HF_TOKEN or token file)", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    repo, rtype = args.repo, "dataset"

    commits = api.list_repo_commits(repo, repo_type=rtype, revision=args.branch)
    files = api.list_repo_files(repo, repo_type=rtype, revision=args.branch)
    print(f"repo {repo}@{args.branch}: {len(commits)} commits, {len(files)} files")

    to_delete: list[str] = []
    if args.run:
        run = args.run.strip("/")
        run_files = [f for f in files if f.startswith(run + "/")]
        if not run_files:
            print(f"ERROR: no files under {run}/", file=sys.stderr)
            return 1
        dirs = sorted({f.split("/")[1] for f in run_files if f.count("/") >= 2})
        to_delete = [f"{run}/{d}" for d in dirs if LEGACY_RE.match(d)]
        print(f"run {run}: {len(run_files)} files; top-level dirs: {dirs}")
        print(f"legacy folders to delete: {to_delete or 'none'}")
    squash = not args.no_squash
    if squash:
        print(f"will squash {len(commits)} commits of {args.branch} into 1 (WHOLE branch, irreversible)")

    if not args.execute:
        print("DRY RUN: nothing changed. Re-run with --execute to apply.")
        return 0

    jobs = running_jobs()
    if jobs and not args.ignore_running_jobs:
        print("ERROR: jobs are queued/running and may push concurrently:\n" + jobs, file=sys.stderr)
        print("Wait for them or pass --ignore-running-jobs.", file=sys.stderr)
        return 2

    if to_delete:
        from huggingface_hub import CommitOperationDelete
        ops = [CommitOperationDelete(path_in_repo=d + "/", is_folder=True) for d in to_delete]
        api.create_commit(repo, ops, repo_type=rtype, revision=args.branch,
                          commit_message=f"Prune legacy checkpoints of {args.run}")
        print(f"deleted {len(to_delete)} legacy folder(s)")
    if squash:
        api.super_squash_history(repo, branch=args.branch, repo_type=rtype,
                                 commit_message="Squash history (scripts/maintenance/squash_hf_history.py)")
        after = api.list_repo_commits(repo, repo_type=rtype, revision=args.branch)
        print(f"squashed: {len(commits)} -> {len(after)} commit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
