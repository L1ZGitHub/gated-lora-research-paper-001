#!/usr/bin/env python3
"""v2 Qwen2.5-0.5B analyses from the small HF result files (no model, no GPU).

  breakdown: per-task answer loss on val (in-distribution held-out) vs test (official splits),
             gated minus r66 / r56, mean over seeds: which tasks carry the val gain, and
             whether it survives on test.
  knockout:  (--knockout DIR, output of knockout_v2.py) does the damage of removing an expert
             follow its routing weight across tasks?
  pertask:   (--pertask DIR) per-task expert removal chosen on val, scored on test
  routing:   Q2 from final_examples.npz (per-example gate means over answer tokens [N, L, E]):
             - expected rank E[r] = sum_e w_e r_e per task x layer (routing map), per seed
             - difficulty -> rank: Spearman across tasks (task loss vs E[r]) and within task
               (example NLL/token vs E[r])
             - task specificity: share of gate variance explained by the task (eta^2), per
               layer, trained gates vs the frozen random gate control
             - reproducibility: correlation of the task x layer maps between seeds

    .venv/bin/python scripts/analysis/analyze_v2.py [--out analysis_v2.json]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "Helain/gated-lora-experiments"
P = "v2_qwen25_05b_"
SEEDS5 = range(5)
ARMS = {  # arm -> (config, seeds, expert ranks or None)
    "gated": ("gated_lr5em5", SEEDS5, (8, 16, 32)),
    "r66": ("baseline_r66_lr5em5", SEEDS5, None),
    "r56": ("baseline_r56", SEEDS5, None),
    "equal_rank": ("equal_rank", range(3), (19, 19, 18)),
    "last_quarter": ("gated_last_quarter", range(3), (8, 16, 32)),
    "frozen_gate": ("frozen_gate", range(3), (8, 16, 32)),
}


def get(run: str, name: str) -> str:
    return hf_hub_download(REPO, f"{P}{run}/{name}", repo_type="dataset")


def spearman(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def breakdown() -> dict:
    per = {}  # arm -> split -> task -> [loss per seed]
    for arm, (cfg, seeds, _) in ARMS.items():
        for s in seeds:
            ev = json.load(open(get(f"{cfg}_seed{s}", "eval_results.json")))
            for split, key in (("val", "val_full"), ("test", "final")):
                for t, m in ev[key]["per_task"].items():
                    per.setdefault(arm, {}).setdefault(split, {}).setdefault(t, []).append(m["answer_loss"])
    tasks = sorted(per["gated"]["val"])
    out = {"tasks": tasks, "mean": {}, "diff": {}}
    for arm in per:
        out["mean"][arm] = {sp: {t: float(np.mean(per[arm][sp][t])) for t in tasks} for sp in ("val", "test")}
    print("\n== Per-task answer loss: gated minus baseline (mean over seeds; negative = gated better)")
    print(f"{'task':14s}" + "".join(f"{b + ' ' + sp:>14s}" for b in ("r66", "r56") for sp in ("val", "test"))
          + f"{'gated val':>11s}{'gated test':>11s}{'seed sd test':>13s}")
    for t in tasks:
        row = []
        for b in ("r66", "r56"):
            for sp in ("val", "test"):
                d = out["mean"]["gated"][sp][t] - out["mean"][b][sp][t]
                out["diff"].setdefault(b, {}).setdefault(sp, {})[t] = d
                row.append(d)
        sd = float(np.std(per["gated"]["test"][t], ddof=1))
        print(f"{t:14s}" + "".join(f"{d:+14.4f}" for d in row)
              + f"{out['mean']['gated']['val'][t]:11.4f}{out['mean']['gated']['test'][t]:11.4f}{sd:13.4f}")
    for b in ("r66", "r56"):
        v, te = (np.mean(list(out["diff"][b][sp].values())) for sp in ("val", "test"))
        print(f"macro mean vs {b}: val {v:+.4f}  test {te:+.4f}")
    return out


def routing() -> dict:
    out = {}
    for arm, (cfg, seeds, ranks) in ARMS.items():
        if ranks is None:
            continue
        r = np.asarray(ranks, float)
        maps, eta_l, etav_l, rho_task, rho_ex = [], [], [], [], []
        for s in seeds:
            z = np.load(get(f"{cfg}_seed{s}", "final_examples.npz"))
            g = z["gate_answer"].astype(np.float32)  # [N, L, E]
            g = g[:, np.isfinite(g).any(axis=(0, 2))]  # gated layers only (last_quarter: 6)
            task = z["task"]
            nll = z["answer_nll_sum"] / np.maximum(z["answer_tokens"], 1)
            tasks = sorted(set(task))
            ok = np.isfinite(g).all(axis=(1, 2)) & (z["answer_tokens"] > 0)
            # permutation-invariant task specificity: between-task share of the total variance
            # of the gate weight vector (summed over experts), per layer
            gm = g[ok].mean(0)
            tot_v = ((g[ok] - gm) ** 2).sum(-1).mean(0)
            btw_v = sum((ok & (task == t)).sum() * ((g[ok & (task == t)].mean(0) - gm) ** 2).sum(-1)
                        for t in tasks) / ok.sum()
            etav_l.append(btw_v / np.maximum(tot_v, 1e-12))
            er = (g * r).sum(-1)  # expected rank [N, L]
            m = np.stack([er[ok & (task == t)].mean(0) for t in tasks])  # [T, L]
            maps.append(m)
            # eta^2 per layer: between-task variance of E[r] / total variance
            tot = er[ok].var(0)
            between = sum((ok & (task == t)).sum() * (er[ok & (task == t)].mean(0) - er[ok].mean(0)) ** 2
                          for t in tasks) / ok.sum()
            eta_l.append(np.where(tot > 0, between / np.maximum(tot, 1e-12), np.nan))
            loss_t = [nll[ok & (task == t)].mean() for t in tasks]
            rho_task.append(spearman(loss_t, m.mean(1)))
            rho_ex.append(np.mean([spearman(nll[ok & (task == t)], er[ok & (task == t)].mean(1))
                                   for t in tasks]))
        maps = np.stack(maps)  # [S, T, L]
        rep = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for a, b in itertools.combinations(maps, 2)]
        eta = np.nanmean(np.stack(eta_l), 0)
        etav = np.mean(np.stack(etav_l), 0)
        L = maps.shape[2]
        thirds = [slice(0, L // 3), slice(L // 3, 2 * L // 3), slice(2 * L // 3, L)]
        out[arm] = {
            "tasks": tasks, "map_mean": maps.mean(0).tolist(), "map_seed_corr": rep,
            "eta2_per_layer": eta.tolist(), "eta2_gate_vector_per_layer": etav.tolist(), "rho_task_difficulty_vs_rank": rho_task,
            "rho_example_difficulty_vs_rank": rho_ex,
        }
        print(f"\n== {arm} ({len(maps)} seeds, expert ranks {ranks})")
        print(f"  task-specificity eta^2 (share of E[rank] variance explained by task), "
              f"layers early/mid/late: " + " / ".join(f"{np.nanmean(eta[s]):.2f}" for s in thirds)
              + f"  (max {np.nanmax(eta):.2f} at layer {int(np.nanargmax(eta))})")
        print(f"  task specificity of the whole gate vector (permutation-invariant), "
              f"layers early/mid/late: " + " / ".join(f"{etav[s].mean():.2f}" for s in thirds))
        print(f"  map reproducibility across seeds (corr of task x layer E[rank]): "
              f"mean {np.mean(rep):.2f} (min {np.min(rep):.2f})")
        print(f"  harder task -> bigger rank? Spearman(task loss, E[rank]) per seed: "
              + " ".join(f"{x:+.2f}" for x in rho_task))
        print(f"  harder example -> bigger rank (within task)? mean Spearman per seed: "
              + " ".join(f"{x:+.2f}" for x in rho_ex))
        mm = maps.mean(0)
        print(f"  E[rank] by task (mean over layers; early/mid/late):")
        for i, t in enumerate(tasks):
            print(f"    {t:14s} {mm[i].mean():5.1f}   " + " / ".join(f"{mm[i, s].mean():5.1f}" for s in thirds))
    return out


def knockout(ko_dir: str) -> dict:
    """Faithfulness: per run, expert e and layer block b, Spearman across tasks between the
    routing weight on e in b (answer tokens) and the test-loss increase when e is removed from b."""
    import glob
    import os
    out = {}
    for f in sorted(glob.glob(os.path.join(ko_dir, "*.json"))):
        k = json.load(open(f))
        run = k["run"]
        z = np.load(get(run, "final_examples.npz"))
        g, task = z["gate_answer"].astype(np.float32), z["task"]
        ok = np.isfinite(g).all(axis=(1, 2)) & (z["answer_tokens"] > 0)
        base = k["conditions"]["none"]["per_task"]
        tasks = sorted(base)
        rows = {}
        for name, c in k["conditions"].items():
            if c["expert"] is None:
                continue
            e, layers = c["expert"], k["blocks"][c["block"]]
            use = [g[ok & (task == t)][:, layers, e].mean() for t in tasks]
            dmg = [c["per_task"][t] - base[t] for t in tasks]
            rows[name] = {"expert": e, "block": c["block"], "macro_damage": float(np.mean(dmg)),
                          "rho_usage_damage": spearman(use, dmg),
                          "damage_per_task": dict(zip(tasks, map(float, dmg)))}
        out[run] = {"intact": k["conditions"]["none"]["mean_task_answer_loss"],
                    "logged": k["logged_final"], "conditions": rows}
    arms = sorted({r.rsplit("_seed", 1)[0] for r in out})
    for arm in arms:
        runs = [r for r in out if r.rsplit("_seed", 1)[0] == arm]
        print(f"\n== knockout {arm} ({len(runs)} seeds): intact vs logged test loss max |d| "
              f"{max(abs(out[r]['intact'] - out[r]['logged']) for r in runs):.1e}")
        print(f"  {'condition':12s} {'macro damage':>13s} {'rho(usage,damage)':>18s}   per seed rho")
        for name in out[runs[0]]["conditions"]:
            dm = [out[r]["conditions"][name]["macro_damage"] for r in runs]
            rh = [out[r]["conditions"][name]["rho_usage_damage"] for r in runs]
            print(f"  {name:12s} {np.mean(dm):+13.4f} {np.nanmean(rh):+18.2f}   "
                  + " ".join(f"{x:+.2f}" for x in rh))
        blk = [out[r]["conditions"][n]["rho_usage_damage"] for r in runs
               for n in out[r]["conditions"] if not n.endswith("_all")]
        print(f"  faithfulness (mean rho over expert x block, all seeds): {np.nanmean(blk):+.2f}")
    return out


def pertask(ko_dir: str) -> dict:
    """Per-task expert removal chosen on VAL, scored on TEST (choosing on test would overstate
    the gain). Input: knockout_v2.py --split val/final --tag _val/_final outputs."""
    import glob
    import os
    base = {}
    for b, (cfg, seeds, _) in (("r66", ARMS["r66"]), ("r56", ARMS["r56"])):
        base[b] = np.mean([json.load(open(get(f"{cfg}_seed{x}", "eval_results.json")))["final"]
                           ["mean_task_answer_loss"] for x in seeds])
    out = {}
    for fv in sorted(glob.glob(os.path.join(ko_dir, "*_val.json"))):
        v = json.load(open(fv))
        f = json.load(open(fv.replace("_val.json", "_final.json")))
        tasks = sorted(v["conditions"]["none"]["per_task"])
        pick = {t: min(v["conditions"], key=lambda c: v["conditions"][c]["per_task"][t]) for t in tasks}
        test_pick = {t: f["conditions"][pick[t]]["per_task"][t] for t in tasks}
        out[v["run"]] = {"pick": pick, "intact": f["conditions"]["none"]["mean_task_answer_loss"],
                         "per_task_policy": float(np.mean(list(test_pick.values()))),
                         "test_per_task_pick": test_pick,
                         "test_per_task_intact": f["conditions"]["none"]["per_task"],
                         "uniform": {c: f["conditions"][c]["mean_task_answer_loss"] for c in f["conditions"]}}
    runs = sorted(out)
    arm = runs[0].rsplit("_seed", 1)[0]
    print(f"\n== Per-task removal chosen on val, scored on test ({arm}, {len(runs)} seeds)")
    print(f"  test macro loss: intact {np.mean([out[r]['intact'] for r in runs]):.4f} | "
          f"val-chosen per task {np.mean([out[r]['per_task_policy'] for r in runs]):.4f} | "
          f"r66 {base['r66']:.4f} | r56 {base['r56']:.4f}")
    print("  same removal for every task (test macro, mean over seeds):")
    for c in out[runs[0]]["uniform"]:
        print(f"    {c:12s} {np.mean([out[r]['uniform'][c] for r in runs]):.4f}")
    tasks = sorted(out[runs[0]]["pick"])
    print("  per task: choice on val in each seed | test change vs intact (mean over seeds)")
    for t in tasks:
        d = np.mean([out[r]["test_per_task_pick"][t] - out[r]["test_per_task_intact"][t] for r in runs])
        print(f"    {t:14s} {' '.join(f'{out[r]['pick'][t]:>10s}' for r in runs)}   {d:+.4f}")
    out["_baselines_test"] = base
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_v2.json")
    ap.add_argument("--routing-only", action="store_true")
    ap.add_argument("--knockout", metavar="DIR", help="only the knockout faithfulness analysis")
    ap.add_argument("--pertask", metavar="DIR", help="only the per-task removal policy analysis")
    args = ap.parse_args()
    if args.pertask:
        res = {"pertask": pertask(args.pertask)}
    elif args.knockout:
        res = {"knockout": knockout(args.knockout)}
    elif args.routing_only:
        res = {"routing": routing()}
    else:
        res = {"breakdown": breakdown(), "routing": routing()}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
