#!/usr/bin/env python3
"""Summarise v2 runs on the HF dataset repo: status, val/final macro answer loss, generation
metrics, trained-gate top-1 sharpness, test loss on the val-selected checkpoint (final*);
and the Phase A decisions (LR per arm, frozen-gate target). Reads only small files (eval_results.json, final_examples.npz); no model.

    .venv/bin/python scripts/analysis/summarize_v2.py [--prefix v2_qwen25_05b] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "Helain/gated-lora-experiments"
LR_SWEEPS = {  # arm -> {lr: config basename}
    "gated": {2.5e-5: "v2_qwen25_05b_gated_lr2p5em5", 5e-5: "v2_qwen25_05b_gated_lr5em5", 1e-4: "v2_qwen25_05b_gated_lr1em4", 2e-4: "v2_qwen25_05b_gated",
              4e-4: "v2_qwen25_05b_gated_lr4em4"},
    "baseline_r66": {2.5e-5: "v2_qwen25_05b_baseline_r66_lr2p5em5",
                     5e-5: "v2_qwen25_05b_baseline_r66_lr5em5", 1e-4: "v2_qwen25_05b_baseline_r66_lr1em4", 2e-4: "v2_qwen25_05b_baseline_r66",
                     4e-4: "v2_qwen25_05b_baseline_r66_lr4em4"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v2_")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    runs = defaultdict(set)
    for f in files:
        m = re.match(rf"({re.escape(args.prefix)}[^/]*_seed\d+)/(.+)$", f)
        if m:
            runs[m.group(1)].add(m.group(2))

    rows = []
    for run in sorted(runs):
        have = runs[run]
        row = {"run": run, "config": run.rsplit("_seed", 1)[0], "seed": int(run.rsplit("_seed", 1)[1]),
               "done": "TRAINING_DONE" in have}
        if not row["done"]:
            st = "latest/training_state.json"
            if st in have:
                s = json.load(open(hf_hub_download(REPO, f"{run}/{st}", repo_type="dataset")))
                row["global_step"] = s.get("global_step")
        if "eval_results.json" in have:
            ev = json.load(open(hf_hub_download(REPO, f"{run}/eval_results.json", repo_type="dataset")))
            row["val_loss"] = ev["val_full"]["mean_task_answer_loss"]
            row["final_loss"] = ev["final"]["mean_task_answer_loss"]
            row["final_em"] = ev["final"].get("mean_task_exact_match")
            row["best_step"] = ev.get("best_eval_step")
            row["generation"] = {t: g["mean"] for t, g in (ev.get("generation") or {}).items()}
            best = ev.get("final_best")  # test split on the val-selected checkpoint (Phase B on)
            if best:
                row["final_loss_best"] = best["final"]["mean_task_answer_loss"]
                row["generation_best"] = {t: g["mean"] for t, g in (best.get("generation") or {}).items()}
        if "final_examples.npz" in have:
            z = np.load(hf_hub_download(REPO, f"{run}/final_examples.npz", repo_type="dataset"))
            if "layer_top1_dominance" in z:
                row["top1"] = float(np.nanmean(z["layer_top1_dominance"]))
        rows.append(row)

    print(f"{'run':52s} {'done':5s} {'step':>5s} {'val':>7s} {'final':>7s} {'EM':>6s} "
          f"{'gsm8k':>6s} {'xsum':>6s} {'top1':>6s} {'best@':>5s} {'final*':>7s}")
    for r in rows:
        g = r.get("generation", {})
        fmt = lambda v, w=7, p=4: f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"
        print(f"{r['run']:52s} {str(r['done']):5s} {str(r.get('global_step', '')):>5s} "
              f"{fmt(r.get('val_loss'))} {fmt(r.get('final_loss'))} {fmt(r.get('final_em'), 6, 3)} "
              f"{fmt(g.get('gsm8k'), 6, 3)} {fmt(g.get('xsum'), 6, 3)} {fmt(r.get('top1'), 6, 3)} "
              f"{str(r.get('best_step', '')):>5s} {fmt(r.get('final_loss_best'))}")

    # Phase A decisions: LR per arm on mean val loss over the seeds that finished
    by_cfg = defaultdict(list)
    for r in rows:
        if r["done"] and "val_loss" in r:
            by_cfg[r["config"]].append(r)
    decisions = {}
    for arm, sweep in LR_SWEEPS.items():
        scores = {lr: float(np.mean([x["val_loss"] for x in by_cfg[c]]))
                  for lr, c in sweep.items() if by_cfg.get(c)}
        n = {lr: len(by_cfg.get(c, [])) for lr, c in sweep.items()}
        if scores:
            best = min(scores, key=scores.get)
            decisions[arm] = {"val_loss_by_lr": scores, "seeds_by_lr": n, "best_lr": best}
            print(f"\n{arm}: val loss by LR {scores} (seeds {n}) -> best LR {best:g}")
    if "gated" in decisions:  # sharpness of the trained gates at the chosen LR
        cfg = LR_SWEEPS["gated"][decisions["gated"]["best_lr"]]
        gated_top1 = [r["top1"] for r in by_cfg.get(cfg, []) if "top1" in r]
        if gated_top1:
            decisions["frozen_gate_target_top1"] = float(np.mean(gated_top1))
            print(f"trained-gate top-1 ({cfg}, {len(gated_top1)} seeds): "
                  f"{np.mean(gated_top1):.3f} -> frozen_gate_target_top1")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"runs": rows, "decisions": decisions}, f, indent=2)


if __name__ == "__main__":
    main()
