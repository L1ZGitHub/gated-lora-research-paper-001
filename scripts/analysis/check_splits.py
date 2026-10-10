#!/usr/bin/env python3
"""Leakage check of the val and final (test) splits against the train pool, per task, on the
real data (CPU, downloads the HF datasets; no tokenizer).

* exact: share of val/test examples whose model input (context + query) also occurs in the
  train pool;
* WikiText: share of val/test paragraphs whose ARTICLE also has paragraphs in the train pool
  (v2 split paragraphs row by row: 100% for val).

    PYTHONPATH=src .venv/bin/python scripts/analysis/check_splits.py --data-format v3
"""

from __future__ import annotations

import argparse

import numpy as np

from gated_lora.data.multi_task_dataset import TASK_SPECS, MultiTaskDatasetLoader


def article_ids(texts) -> np.ndarray:
    """Article index per WikiText row: an article starts at a top-level header ' = Title = '."""
    ids, cur = [], -1
    for t in texts:
        s = t.strip()
        if s.startswith("= ") and s.endswith(" =") and not s.startswith("= ="):
            cur += 1
        ids.append(cur)
    return np.array(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-format", default="v3", choices=["v2", "v3"])
    ap.add_argument("--tasks", nargs="+", default=list(TASK_SPECS))
    args = ap.parse_args()
    ld = MultiTaskDatasetLoader.__new__(MultiTaskDatasetLoader)  # split logic only, no tokenizer
    ld.__dict__.update(data_format=args.data_format, split_seed=0, seed=0, val_fraction=0.05,
                       _partition_cache={}, _hf_cache={})
    print(f"data_format {args.data_format}: share of examples whose input occurs in train")
    for task in args.tasks:
        spec = TASK_SPECS[task]
        tr_ds, tr_idx = ld._candidates(task, "train")
        inp = lambda e: e.context + e.query  # noqa: E731
        train_inputs = {inp(e) for e in map(spec.formatter, tr_ds.select(tr_idx.tolist())) if e}
        for role in ("val", "final"):
            ds, idx = ld._candidates(task, role)
            exs = [e for e in map(spec.formatter, ds.select(idx.tolist())) if e]
            exact = np.mean([inp(e) in train_inputs for e in exs])
            line = (f"  {task:14s} {role:5s} ({ld.hf_split(task, role):18s}) n={len(exs):6d} "
                    f"exact {exact:6.1%}")
            if task == "wikitext":
                if ds is tr_ds:  # same split: article-level overlap
                    art = article_ids(ds["text"])
                    train_art = set(art[tr_idx].tolist())
                    rows = [i for i in idx if spec.formatter(ds[int(i)])]
                    line += f"  same article {np.mean([art[i] in train_art for i in rows]):6.1%}"
                else:
                    line += "  same article   0.0% (different official split)"
            print(line, flush=True)


if __name__ == "__main__":
    main()
