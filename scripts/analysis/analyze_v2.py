#!/usr/bin/env python3
"""Qwen2.5-0.5B analyses of v2 or v3 runs (--version) from the small HF result files (no model,
no GPU).

  q1:        pre-registered Q1 test (experiments/README.md, v3): per-seed macro test loss (last
             checkpoint), gated vs each plain LoRA, Welch t-test with Holm correction, 95% CIs;
             plus the robustness metric (loss / base-model loss, v3_base_reference.json)
  capacity:  (v3small) pre-registered "capacity-bound at rank X" test: plain rank X vs the
             Phase B' r16, Welch + Holm, and mean test loss above r16 by more than DELTA

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

    .venv/bin/python scripts/analysis/analyze_v2.py [--version v2] [--gates-dir DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "Helain/gated-lora-experiments"
SEEDS5 = range(5)
VERSIONS = {  # version -> arm -> (config, seeds, expert ranks or None)
    "v2": {
        "gated": ("gated_lr5em5", SEEDS5, (8, 16, 32)),
        "r66": ("baseline_r66_lr5em5", SEEDS5, None),
        "r56": ("baseline_r56", SEEDS5, None),
        "r32": ("baseline_r32", SEEDS5, None),
        "r16": ("baseline_r16", SEEDS5, None),
        "equal_rank": ("equal_rank", range(3), (19, 19, 18)),
        "last_quarter": ("gated_last_quarter", range(3), (8, 16, 32)),
        "frozen_gate": ("frozen_gate", range(3), (8, 16, 32)),
    },
    "v3": {  # LR per arm from the 9-point Phase A' grid
        "gated": ("gated_lr2p5em5", SEEDS5, (8, 16, 32)),
        "r66": ("baseline_r66_lr2p5em5", SEEDS5, None),
        "r56": ("baseline_r56", SEEDS5, None),
        "r32": ("baseline_r32_lr5em5", SEEDS5, None),
        "r16": ("baseline_r16_lr1em4", SEEDS5, None),
        "equal_rank": ("equal_rank", range(3), (19, 19, 18)),
        "frozen_gate": ("frozen_gate", range(3), (8, 16, 32)),
    },
    "v3small": {  # Phase C': LR per arm from the 9-point small-rank grid; r16/r32 = Phase B' runs
        "gated": ("gated_r2r4r8_lr1em4", SEEDS5, (2, 4, 8)),
        "r14": ("baseline_r14_lr1em4", SEEDS5, None),
        "r8": ("baseline_r8_lr1em4", SEEDS5, None),
        "r4": ("baseline_r4_lr2p5em4", SEEDS5, None),
        "r2": ("baseline_r2_lr2p5em4", SEEDS5, None),
        "r16": ("baseline_r16_lr1em4", SEEDS5, None),
        "r32": ("baseline_r32_lr5em5", SEEDS5, None),
    },
}
SECONDARY = {"v3small": ("r16", "r32")}  # reported with CIs, outside the Holm family
CAPACITY = {"v3small": ("r16", ("r14", "r8", "r4", "r2")),  # reference arm, arms tested against it
            "v3smollm": ("r56", ("r14", "r2"))}
MODELS = {"v3smollm": "smollm360m"}  # version -> model tag of the run names (default qwen25_05b)
# Q1 reference arm per plain arm when a version has two gated models (default: "gated")
Q1_REF = {"v3smollm": {"r14": "gated_small", "r2": "gated_small"}}
TWIN = {"gated": "r56", "gated_small": "r14"}  # Phase D: routing-off twin of each gated model
DELTA = 0.005  # Q1 equivalence margin (macro test loss, nats; README v3 amendment)
P, ARMS = "v3_qwen25_05b_", VERSIONS["v3"]  # set by --version
VERSION = "v3"
GATES_DIR = None  # --gates-dir: re-dumped per-example files (knockout_v2.py --dump-dir)


def get(run: str, name: str) -> str:
    return hf_hub_download(REPO, f"{P}{run}/{name}", repo_type="dataset")


def examples(run: str):
    """Per-example test dump of a run. v3 requires the re-dumped files (gate means at the
    positions predicting answer tokens); the training-time final_examples.npz of v2 and early v3
    runs averaged the answer tokens themselves (one position late)."""
    if GATES_DIR:
        return np.load(f"{GATES_DIR}/{P}{run}_final_examples.npz")
    z = np.load(get(run, "final_examples.npz"))
    if "gate_positions" not in z and not P.startswith("v2_"):
        raise SystemExit(f"{run}: final_examples.npz has the old gate positions; re-dump with "
                         "knockout_v2.py --conditions none --dump-dir DIR and pass --gates-dir DIR")
    return z


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
    vs = [b for b in ARMS if ARMS[b][2] is None][:2]  # v3: r66, r56; v3small: r14, r8
    out = {"tasks": tasks, "mean": {}, "diff": {}}
    for arm in per:
        out["mean"][arm] = {sp: {t: float(np.mean(per[arm][sp][t])) for t in tasks} for sp in ("val", "test")}
    print("\n== Per-task answer loss: gated minus baseline (mean over seeds; negative = gated better)")
    print(f"{'task':14s}" + "".join(f"{b + ' ' + sp:>14s}" for b in vs for sp in ("val", "test"))
          + f"{'gated val':>11s}{'gated test':>11s}{'seed sd test':>13s}")
    for t in tasks:
        row = []
        for b in vs:
            for sp in ("val", "test"):
                d = out["mean"]["gated"][sp][t] - out["mean"][b][sp][t]
                out["diff"].setdefault(b, {}).setdefault(sp, {})[t] = d
                row.append(d)
        sd = float(np.std(per["gated"]["test"][t], ddof=1))
        print(f"{t:14s}" + "".join(f"{d:+14.4f}" for d in row)
              + f"{out['mean']['gated']['val'][t]:11.4f}{out['mean']['gated']['test'][t]:11.4f}{sd:13.4f}")
    for b in vs:
        v, te = (np.mean(list(out["diff"][b][sp].values())) for sp in ("val", "test"))
        print(f"macro mean vs {b}: val {v:+.4f}  test {te:+.4f}")
    return out


def holm(p: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    adj, run_max = {}, 0.0
    for i, k in enumerate(sorted(p, key=p.get)):
        run_max = max(run_max, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run_max
    return adj


def q1(base_ref: str | None) -> dict:
    """Pre-registered Q1 test: per-seed macro test loss (plain mean over tasks, last checkpoint),
    gated vs each plain LoRA arm; Welch t-test, two-sided, Holm over the comparisons. "Better"
    needs Holm-adjusted p < 0.05 and a lower mean. Equivalence (amendment 2026-10-11): TOST with
    margin DELTA, Holm over the comparisons. Robustness: superiority on loss / base loss."""
    from scipy import stats  # installed with scikit-learn

    per = {arm: [json.load(open(get(f"{cfg}_seed{s}", "eval_results.json")))["final"]["per_task"]
                 for s in seeds] for arm, (cfg, seeds, _) in ARMS.items()}
    ref = Q1_REF.get(VERSION, {})
    secondary = [b for b in per if b in SECONDARY.get(VERSION, ())]
    others = [b for b in per if ARMS[b][2] is None and b not in secondary]
    tasks = sorted(per["gated"][0])
    print("\n== Per-task test answer loss (last checkpoint), mean +- sd over seeds")
    print(f"{'arm':12s}" + "".join(f"{t[:11]:>16s}" for t in tasks))
    for a, rows in per.items():
        v = np.array([[pt[t]["answer_loss"] for t in tasks] for pt in rows])
        sd = v.std(0, ddof=1) if len(v) > 1 else np.full(len(tasks), np.nan)
        print(f"{a:12s}" + "".join(f"{m:9.4f}+-{e:.4f}"[-16:].rjust(16) for m, e in zip(v.mean(0), sd, strict=True)))
    out = {"tasks": tasks, "per_task": {a: [[pt[t]["answer_loss"] for t in tasks] for pt in rows]
                                        for a, rows in per.items()}}
    metrics = [("raw", lambda t: 1.0)]
    if base_ref:
        base = json.load(open(base_ref))["final"]
        metrics.append(("rel_to_base", lambda t: 1.0 / base[t]))
    for metric, w in metrics:
        macro = {a: np.array([np.mean([m["answer_loss"] * w(t) for t, m in pt.items()]) for pt in rows])
                 for a, rows in per.items()}
        res = {}
        for b in others + secondary:
            x, g = macro[b], macro[ref.get(b, "gated")]
            d = g.mean() - x.mean()
            vg, vx = g.var(ddof=1) / len(g), x.var(ddof=1) / len(x)
            df = (vg + vx) ** 2 / (vg ** 2 / (len(g) - 1) + vx ** 2 / (len(x) - 1))
            half = stats.t.ppf(0.975, df) * np.sqrt(vg + vx)
            se = np.sqrt(vg + vx)
            half90 = stats.t.ppf(0.95, df) * se
            # the margin is defined on the primary (raw) metric only
            p_tost = (max(stats.t.sf((d + DELTA) / se, df), stats.t.cdf((d - DELTA) / se, df))
                      if metric == "raw" else float("nan"))
            res[b] = {"diff": float(d), "ci95": [float(d - half), float(d + half)],
                      "ci90": [float(d - half90), float(d + half90)], "df": float(df),
                      "p": float(stats.ttest_ind(g, x, equal_var=False).pvalue),
                      "p_tost": float(p_tost),
                      "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n": len(x)}
            if ref:
                res[b]["ref"] = ref.get(b, "gated")
        p_holm = holm({b: res[b]["p"] for b in others})
        p_tost_holm = (holm({b: res[b]["p_tost"] for b in others}) if metric == "raw"
                       else {b: float("nan") for b in others})
        for b in secondary:  # pre-registered as CIs only: no test, no verdict
            res[b]["p_holm"] = res[b]["p_tost_holm"] = float("nan")
            res[b]["verdict"] = "secondary (CI only)"
        for b in others:
            res[b]["p_holm"], res[b]["p_tost_holm"] = p_holm[b], p_tost_holm[b]
            sup = ("gated better" if p_holm[b] < 0.05 and res[b]["diff"] < 0 else
                   "gated worse" if p_holm[b] < 0.05 else "")
            eq = f"equivalent within {DELTA}" if p_tost_holm[b] < 0.05 else ""
            # no equivalence test on the robustness metric: a non-significant difference there
            # is "no detectable difference", not a failed equivalence
            res[b]["verdict"] = ("; ".join(v for v in (sup, eq) if v)
                                 or ("inconclusive" if metric == "raw" else "no detectable difference"))
        g = macro["gated"]
        out[metric] = {"gated": {"mean": float(g.mean()), "sd": float(g.std(ddof=1)), "n": len(g)},
                       "vs": res, "per_seed": {a: v.tolist() for a, v in macro.items()}}
        print(f"\n== Q1 ({metric}{', primary' if metric == 'raw' else ', robustness'}): "
              f"macro test loss, gated {g.mean():.4f} +- {g.std(ddof=1):.4f} (n={len(g)})")
        print(f"  {'vs':5s} {'mean':>8s} {'sd':>7s} {'gated-x':>9s} {'95% CI (unadj.)':>21s} "
              f"{'p':>6s} {'p_holm':>6s} {'90% CI':>21s} {'p_TOST':>6s} {'holm':>6s}  verdict")
        for b in others + secondary:
            r = res[b]
            print(f"  {b:5s} {r['mean']:8.4f} {r['sd']:7.4f} {r['diff']:+9.4f} "
                  + (f"({r['ref']}) " if ref else "") +
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {r['p']:6.3f} {r['p_holm']:6.3f} "
                  f"[{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}] {r['p_tost']:6.3f} "
                  f"{r['p_tost_holm']:6.3f}  {r['verdict']}")
    return out


def capacity() -> dict:
    """Pre-registered Phase C' test (README, 2026-10-11): plain rank X is "capacity-bound" if its
    macro test loss differs from the reference arm (Welch, two-sided, Holm over the tested arms,
    p < 0.05) and its mean exceeds the reference mean by more than DELTA."""
    from scipy import stats

    ref, arms = CAPACITY[VERSION]
    macro = {a: np.array([json.load(open(get(f"{ARMS[a][0]}_seed{s}", "eval_results.json")))
                          ["final"]["mean_task_answer_loss"] for s in ARMS[a][1]])
             for a in (ref, *arms)}
    r = macro[ref]
    p_holm = holm({a: float(stats.ttest_ind(macro[a], r, equal_var=False).pvalue) for a in arms})
    out = {"reference": ref, "reference_mean": float(r.mean()), "vs": {}}
    print(f"\n== Capacity-bound test vs {ref} (macro test loss {r.mean():.4f} +- {r.std(ddof=1):.4f})")
    print(f"  {'arm':5s} {'mean':>8s} {'sd':>7s} {'x-' + ref:>9s} {'p_holm':>7s}  capacity-bound")
    for a in arms:
        x = macro[a]
        d = float(x.mean() - r.mean())
        bound = bool(p_holm[a] < 0.05 and d > DELTA)
        out["vs"][a] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n": len(x),
                        "diff": d, "p_holm": p_holm[a], "capacity_bound": bound}
        print(f"  {a:5s} {x.mean():8.4f} {x.std(ddof=1):7.4f} {d:+9.4f} {p_holm[a]:7.3f}  {bound}")
    if not any(v["capacity_bound"] for v in out["vs"].values()):
        print("  no arm qualifies: the capacity-scarce regime was not reached")
    return out


def routing(base_ref: str | None = None) -> dict:
    out = {}
    base = json.load(open(base_ref))["final"] if base_ref else None
    for arm, (cfg, seeds, ranks) in ARMS.items():
        if ranks is None:
            continue
        r = np.asarray(ranks, float)
        maps, shares, eta_l, etav_l, rho_task, rho_ex = [], [], [], [], [], []
        raw, rho_gain = [], []  # uncentred shares [T, L, E]; largest-expert share vs relative gain
        for s in seeds:
            z = examples(f"{cfg}_seed{s}")
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
            sh = np.stack([g[ok & (task == t)].mean(0) for t in tasks])  # [T, L, E] gate share
            shares.append(sh - sh.mean(0))  # centred over tasks: shared depth profile removed
            raw.append(sh)
            # eta^2 per layer: between-task variance of E[r] / total variance
            tot = er[ok].var(0)
            between = sum((ok & (task == t)).sum() * (er[ok & (task == t)].mean(0) - er[ok].mean(0)) ** 2
                          for t in tasks) / ok.sum()
            eta_l.append(np.where(tot > 0, between / np.maximum(tot, 1e-12), np.nan))
            loss_t = [nll[ok & (task == t)].mean() for t in tasks]
            if base:  # relative loss reduction from fine-tuning, per task
                rho_gain.append(spearman(sh[:, :, int(np.argmax(r))].mean(1),
                                         [1 - lt / base[t] for lt, t in zip(loss_t, tasks, strict=True)]))
            rho_task.append(spearman(loss_t, m.mean(1)))
            rho_ex.append(np.mean([spearman(nll[ok & (task == t)], er[ok & (task == t)].mean(1))
                                   for t in tasks]))
        maps = np.stack(maps)  # [S, T, L]
        # reproducibility (amendment 2026-10-11): task-centred gate-share maps; experts of equal
        # rank have no identity, so each layer's experts are matched across the two seeds by the
        # best permutation (conservative: only helps that control); mixed ranks: identity
        n_exp = len(ranks)
        perms = (list(itertools.permutations(range(n_exp))) if len(set(ranks)) < n_exp
                 else [tuple(range(n_exp))])
        rep = []
        for a, b in itertools.combinations(shares, 2):
            bm = np.stack([max((b[:, li][:, list(pm)] for pm in perms),
                               key=lambda x, li=li: np.corrcoef(a[:, li].ravel(), x.ravel())[0, 1])
                           for li in range(a.shape[1])], axis=1)
            rep.append(float(np.corrcoef(a.ravel(), bm.ravel())[0, 1]))
        # depth profile (shares averaged over tasks, [L, E]): seed-to-seed correlation
        prof = [x.mean(0) for x in raw]
        prof_rep = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
                    for a, b in itertools.combinations(prof, 2)]
        eta = np.nanmean(np.stack(eta_l), 0)
        etav = np.mean(np.stack(etav_l), 0)
        L = maps.shape[2]
        thirds = [slice(0, L // 3), slice(L // 3, 2 * L // 3), slice(2 * L // 3, L)]
        out[arm] = {
            "tasks": tasks, "map_mean": maps.mean(0).tolist(), "map_seed_corr": rep,
            "eta2_per_layer": eta.tolist(), "eta2_gate_vector_per_layer": etav.tolist(), "rho_task_difficulty_vs_rank": rho_task,
            "rho_example_difficulty_vs_rank": rho_ex,
            "share_mean": np.mean(raw, 0).tolist(), "profile_seed_corr": prof_rep,
            "rho_largest_share_vs_relative_gain": rho_gain,
        }
        print(f"\n== {arm} ({len(maps)} seeds, expert ranks {ranks})")
        print(f"  task-specificity eta^2 (share of E[rank] variance explained by task), "
              f"layers early/mid/late: " + " / ".join(f"{np.nanmean(eta[s]):.2f}" for s in thirds)
              + f"  (max {np.nanmax(eta):.2f} at layer {int(np.nanargmax(eta))})")
        print(f"  task specificity of the whole gate vector (permutation-invariant), "
              f"layers early/mid/late: " + " / ".join(f"{etav[s].mean():.2f}" for s in thirds))
        print(f"  map reproducibility across seeds (corr of task-centred task x layer x expert "
              f"gate shares{', experts matched per layer' if len(perms) > 1 else ''}): "
              f"mean {np.mean(rep):.2f} (min {np.min(rep):.2f})")
        print(f"  harder task -> bigger rank? Spearman(task loss, E[rank]) per seed: "
              + " ".join(f"{x:+.2f}" for x in rho_task))
        print(f"  harder example -> bigger rank (within task)? mean Spearman per seed: "
              + " ".join(f"{x:+.2f}" for x in rho_ex))
        sm = np.mean(raw, 0)  # [T, L, E]
        fmt = lambda v: "/".join(f"{x:.2f}" for x in v)  # noqa: E731
        print(f"  depth-profile reproducibility (corr of layer x expert shares between seeds): "
              f"mean {np.mean(prof_rep):.2f} (min {np.min(prof_rep):.2f})")
        if rho_gain:
            print("  Spearman(largest-expert share, relative loss reduction) per seed: "
                  + " ".join(f"{x:+.2f}" for x in rho_gain))
        print(f"  gate shares (experts by rank): all {fmt(sm.mean((0, 1)))} | early/mid/late "
              + " | ".join(fmt(sm[:, s].mean((0, 1))) for s in thirds))
        for i, t in enumerate(tasks):
            print(f"    {t:14s} " + " | ".join(fmt(sm[i, s].mean(0)) for s in thirds))
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
        if k.get("split", "final") != "final" or k.get("version", "v2") != P[:2]:
            continue  # val-split files are for pertask(); other versions never mix in
        run = k["run"]
        z = examples(run)
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
    for b, (cfg, seeds, _) in ((b, a) for b, a in ARMS.items() if a[2] is None):
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
          + " | ".join(f"{b} {v:.4f}" for b, v in base.items()))
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


def replication(analysis: str, ko: str) -> dict:
    """Phase D verdicts R1-R7 exactly as pre-registered (experiments/README.md, 2026-10-12), from
    the saved full analysis (q1, capacity, routing) and the saved knockout analysis."""
    a, k = json.load(open(analysis)), json.load(open(ko))["knockout"]
    mixes = {arm: v for arm, v in ARMS.items() if v[2] is not None}  # gated, gated_small
    vs, out = a["q1"]["raw"]["vs"], {}

    def cond(arm: str, name: str, key: str) -> list:
        cfg, seeds, _ = ARMS[arm]
        return [k[f"{cfg}_seed{s}"]["conditions"][name][key] for s in seeds]

    out["R1"] = {"ok": all(vs[b]["p_tost_holm"] < 0.05 for b in TWIN.values()),
                 "detail": {f"{g} vs {b}": vs[b]["p_tost_holm"] for g, b in TWIN.items()}}
    out["R2"] = {"ok": not a["capacity"]["vs"]["r2"]["capacity_bound"],
                 "detail": {x: a["capacity"]["vs"]["r2"][x] for x in ("diff", "p_holm")}}
    r3, r4, r5, r7 = {}, {}, {}, {}
    for arm, (_, _, ranks) in mixes.items():
        big, small = int(np.argmax(ranks)), int(np.argmin(ranks))
        dmg = {e: float(np.mean(cond(arm, f"ko{e}_all", "macro_damage"))) for e in range(len(ranks))}
        r3[arm] = {"damage": dmg, "ok": bool(
            dmg[big] >= 10 * max(abs(d) for e, d in dmg.items() if e != big))}
        rho = [x for b in ("early", "mid", "late") for x in cond(arm, f"ko{big}_{b}", "rho_usage_damage")]
        r4[arm] = {"mean": float(np.mean(rho)), "positive": int(np.sum(np.array(rho) > 0)),
                   "n": len(rho), "ok": bool(np.mean(rho) > 0 and np.sum(np.array(rho) > 0) >= 12)}
        rep = float(np.mean(a["routing"][arm]["map_seed_corr"]))
        r5[arm] = {"mean": rep, "ok": rep >= 0.5}
        r7[arm] = {"damage": dmg[small], "ok": abs(dmg[small]) < DELTA}
    for name, r in (("R3", r3), ("R4", r4), ("R5", r5), ("R7", r7)):
        out[name] = {"ok": all(v["ok"] for v in r.values()), "detail": r}
    rho = a["routing"]["gated_small"]["rho_task_difficulty_vs_rank"]
    out["R6"] = {"ok": bool(max(rho) < 0 and np.mean(rho) <= -0.7),
                 "detail": {"per_seed": rho, "mean": float(np.mean(rho))}}
    print("\n== Phase D replication criteria")
    for name in sorted(out):
        print(f"  {name}: {'replicated' if out[name]['ok'] else 'NOT replicated'}  "
              f"{json.dumps(out[name]['detail'])}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v3")
    ap.add_argument("--out", default=None, help="default analysis_<version>.json")
    ap.add_argument("--base-ref", default=None,
                    help="base-model loss per task (scripts/analysis/base_reference.py); "
                         "default experiments/v3_base_reference.json for v3, none for v2")
    ap.add_argument("--gates-dir", default=None,
                    help="re-dumped per-example files (knockout_v2.py --dump-dir); required "
                         "for v3 routing/knockout analyses")
    ap.add_argument("--q1-only", action="store_true")
    ap.add_argument("--routing-only", action="store_true")
    ap.add_argument("--knockout", metavar="DIR", help="only the knockout faithfulness analysis")
    ap.add_argument("--pertask", metavar="DIR", help="only the per-task removal policy analysis")
    ap.add_argument("--replication", nargs=2, metavar=("ANALYSIS", "KNOCKOUT"),
                    help="only the Phase D verdicts R1-R7, from the two saved analysis files")
    args = ap.parse_args()
    global P, ARMS, GATES_DIR, VERSION
    VERSION = args.version
    P, ARMS = f"{args.version[:2]}_{MODELS.get(args.version, 'qwen25_05b')}_", VERSIONS[args.version]
    GATES_DIR = args.gates_dir
    if args.base_ref is None and args.version.startswith("v3"):
        m = MODELS.get(args.version)
        args.base_ref = f"experiments/v3_base_reference{'_' + m if m else ''}.json"
    args.out = args.out or f"analysis_{args.version}.json"
    if args.q1_only:
        res = {"q1": q1(args.base_ref)}
        if VERSION in CAPACITY:
            res["capacity"] = capacity()
    elif args.replication:
        res = {"replication": replication(*args.replication)}
    elif args.pertask:
        res = {"pertask": pertask(args.pertask)}
    elif args.knockout:
        res = {"knockout": knockout(args.knockout)}
    elif args.routing_only:
        res = {"routing": routing(args.base_ref)}
    else:
        res = {"q1": q1(args.base_ref), "routing": routing(args.base_ref)}
        if VERSION not in Q1_REF:  # the gated-minus-baseline breakdown assumes one gated arm
            res["breakdown"] = breakdown()
        if VERSION in CAPACITY:
            res["capacity"] = capacity()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
