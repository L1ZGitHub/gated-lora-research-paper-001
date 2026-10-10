#!/usr/bin/env python3
"""Paper figures for the v2 Qwen2.5-0.5B experiments (PDF + PNG), from the small HF result files
and the knockout outputs of knockout_v2.py. Light mode only (print). Colours: the dataviz
reference palette (categorical slot 1 blue for the gated arm, a neutral for context arms;
sequential blue ramp; diverging blue <-> red around a gray midpoint).

    .venv/bin/python scripts/analysis/figures_v2.py --ko-dir ~/GatedLoraProject/ko_results --out figures

fig1_q1_losses   val and test answer loss per arm (seeds + mean +- sd), incl. the rank curve
fig2_routing     gate share per task x layer for the rank-8 / 16 / 32 experts (5 seeds)
fig3_knockout    test-loss change per task when an expert is removed from a layer block
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Helain/gated-lora-experiments"
P = "v2_qwen25_05b_"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, NEUTRAL = "#2a78d6", "#6e6d69"
SEQ = ["#f4f8fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV = ["#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9b2423"]
TASK_ORDER = ["imdb", "mnli", "conll2003", "commonsenseqa", "squad", "gsm8k", "wikitext", "xsum"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titlecolor": INK,
    "axes.linewidth": 0.8, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42,
})


def get(run: str, name: str) -> str:
    return hf_hub_download(REPO, f"{P}{run}/{name}", repo_type="dataset")


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}.pdf/.png")


def fig1(out: Path) -> None:
    arms = [  # label, config, seeds, emphasised
        ("Gated 8/16/32", "gated_lr5em5", range(5), True),
        ("LoRA r66 (param-matched)", "baseline_r66_lr5em5", range(5), False),
        ("LoRA r56 (= routing off)", "baseline_r56", range(5), False),
        ("Gated, equal ranks 19/19/18", "equal_rank", range(3), False),
        ("Gated, last quarter only", "gated_last_quarter", range(3), False),
        ("Gated, one global gate", "global_gate", range(1), False),
        ("Frozen random gate", "frozen_gate", range(3), False),
        ("LoRA r32", "baseline_r32", range(5), False),
        ("LoRA r16", "baseline_r16", range(5), False),
        ("LoRA r8", "baseline_r8", range(1), False),
    ]
    vals = {}
    for label, cfg, seeds, _ in arms:
        ev = [json.load(open(get(f"{cfg}_seed{s}", "eval_results.json"))) for s in seeds]
        vals[label] = {"val": [e["val_full"]["mean_task_answer_loss"] for e in ev],
                       "test": [e["final"]["mean_task_answer_loss"] for e in ev]}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, split, title in ((axes[0], "val", "Validation (held out from training data)"),
                             (axes[1], "test", "Test (official splits)")):
        ref = np.mean(vals[arms[0][0]][split])
        ax.axvline(ref, color=BLUE, lw=0.8, ls=(0, (3, 3)), zorder=1)
        for yi, (label, _, seeds, emph) in zip(y, arms):
            v = np.array(vals[label][split])
            c = BLUE if emph else NEUTRAL
            ax.scatter(v, np.full(len(v), yi), s=14, facecolors="none", edgecolors=c, lw=0.8, zorder=2)
            if len(v) > 1:
                ax.errorbar(v.mean(), yi, xerr=v.std(ddof=1), fmt="o", ms=5.5, color=c,
                            mec=SURF, mew=1.0, elinewidth=1.2, capsize=0, zorder=3)
            else:
                ax.plot(v.mean(), yi, "D", ms=5, color=c, mec=SURF, mew=1.0, zorder=3)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel("Mean answer loss over 8 tasks (lower is better)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{a[0]}  (n={len(a[2])})" for a in arms])
    for t, a in zip(axes[0].get_yticklabels(), arms):
        t.set_color(INK if a[3] else INK2)
    fig.text(0.0, -0.10, "Hollow circles: seeds. Filled: mean ± 1 s.d. Diamond: single seed. "
             "Dashed line: gated mean. LR 5e-5 for every arm.", color=INK2, fontsize=7)
    save(fig, out, "fig1_q1_losses")


def routing_shares() -> tuple[list[str], np.ndarray]:
    maps = []
    for s in range(5):
        z = np.load(get(f"gated_lr5em5_seed{s}", "final_examples.npz"))
        g, t = z["gate_answer"].astype(np.float32), z["task"]
        maps.append(np.stack([g[t == k].mean(0) for k in TASK_ORDER]))  # [T, L, E]
    return TASK_ORDER, np.mean(maps, 0)


def fig2(out: Path) -> None:
    tasks, m = routing_shares()
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    for e, (ax, name) in enumerate(zip(axes, ("rank-8 expert", "rank-16 expert", "rank-32 expert"))):
        im = ax.imshow(m[:, :, e], cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_title(name, loc="left", pad=5)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Layer")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(tasks)))
    axes[0].set_yticklabels(tasks)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Mean gate weight on answer tokens", color=INK2)
    cb.ax.tick_params(length=0)
    fig.suptitle("Share of gate weight per task and layer", x=0.0, ha="left", y=1.04,
                 fontsize=8.5, color=INK)
    fig.text(0.0, -0.08, "Gated 8/16/32 model, test split, answer tokens, mean of 5 seeds. "
             "For each cell, the three panels sum to 1.", color=INK2, fontsize=7)
    save(fig, out, "fig2_routing")


def fig3(out: Path, ko_dir: str) -> None:
    ko = [json.load(open(os.path.join(ko_dir, f"gated_lr5em5_seed{s}.json"))) for s in range(5)]
    cols = [(e, b) for e in range(3) for b in ("early", "mid", "late", "all")]
    d = np.array([[np.mean([k["conditions"][f"ko{e}_{b}"]["per_task"][t]
                            - k["conditions"]["none"]["per_task"][t] for k in ko])
                   for e, b in cols] for t in TASK_ORDER])
    lim = float(np.abs(d).max())
    norm = SymLogNorm(linthresh=0.005, linscale=0.6, vmin=-lim, vmax=lim, base=10)
    cmap = LinearSegmentedColormap.from_list("div", DIV)
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    ax.imshow(d, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d[i, j]
            dark = abs(v) > 0.05
            ax.text(j, i, "0.000" if abs(v) < 5e-4 else f"{v:+.3f}", ha="center", va="center", fontsize=6.3,
                    color=SURF if dark else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([b for _, b in cols])
    for j in (3.5, 7.5):
        ax.axvline(j, color=SURF, lw=3)
    for e, name in enumerate(("rank-8 removed", "rank-16 removed", "rank-32 removed")):
        ax.text(e * 4 + 1.5, -0.95, name, ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER)
    ax.set_xlabel("Layers where the expert is removed (early 0-7, mid 8-15, late 16-23)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.0, -0.07, "Change in test answer loss vs the intact model (red: worse, blue: better), "
             "mean of 5 seeds; colour on a symmetric log scale.", color=INK2, fontsize=7)
    save(fig, out, "fig3_knockout")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ko-dir", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(out)
    fig3(out, args.ko_dir)


if __name__ == "__main__":
    main()
