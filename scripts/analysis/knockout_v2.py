#!/usr/bin/env python3
"""Expert knockout on trained v2 gated runs (GPU). For each run: rebuild the model exactly as in
training (same config + seed), load <run>/final_model from HF, check the intact model reproduces
the logged test loss, then evaluate the full test split with expert e's gate weight set to 0
(no renormalisation: its contribution is removed, the others are unchanged) in all layers or
in one third of the layers. Faithfulness (does damage follow the routing weights?) is computed
by `analyze_v2.py --knockout`.

    PYTHONPATH=src .venv/bin/python scripts/analysis/knockout_v2.py \
        --runs gated_lr5em5_seed0 frozen_gate_seed0 --out-dir ko_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from gated_lora.cli import dict_to_experiment_config
from gated_lora.training import load_config
from gated_lora.training.gated_trainer import GatedLoRATrainer, create_optimizer_and_scheduler
from gated_lora.training.config import resolve_precision
from gated_lora.training.pipeline import build_data, build_model

REPO = "Helain/gated-lora-experiments"
P = "v2_qwen25_05b_"
logger = logging.getLogger("knockout")


def run_knockout(run: str, out_dir: Path) -> None:
    cfg_name, seed = run.rsplit("_seed", 1)
    out_file = out_dir / f"{run}.json"
    if out_file.exists():
        logger.info(f"{run}: done already, skip")
        return
    work = Path(tempfile.mkdtemp(prefix=f"ko_{run}_"))
    config = dict_to_experiment_config(load_config(f"configs/experiments/v2/{P}{cfg_name}.yaml"))
    config.seed = config.training.seed = int(seed)
    config.output_dir = str(work / "out")
    config.training.push_to_hub = False
    config.wandb.enabled = False
    config.validate()
    Path(config.output_dir).mkdir(parents=True)

    model, tokenizer = build_model(config)
    data = build_data(config, tokenizer)
    opt, sched = create_optimizer_and_scheduler(model=model, config=config, num_training_steps=1)
    trainer = GatedLoRATrainer(model=model, data=data, optimizer=opt, scheduler=sched, config=config,
                               output_dir=config.output_dir, device="cuda", max_steps=1,
                               precision=resolve_precision(config.model.precision))
    snap = snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{P}{run}/final_model/*"],
                             local_dir=str(work / "hf"))
    trainer.load_weights(Path(snap) / f"{P}{run}" / "final_model")
    logged = json.load(open(snapshot_download(
        REPO, repo_type="dataset", allow_patterns=[f"{P}{run}/eval_results.json"],
        local_dir=str(work / "hf")) + f"/{P}{run}/eval_results.json"))["final"]

    gn = model.gating_network
    orig = gn.compute_gate_weights
    state = {"expert": None, "layers": set()}

    def patched(hidden_states, layer_idx=0, *a, **kw):
        gw, gl, info = orig(hidden_states, layer_idx, *a, **kw)
        if state["expert"] is not None and layer_idx in state["layers"]:
            gw = gw.clone()
            gw[..., state["expert"]] = 0
        return gw, gl, info

    gn.compute_gate_weights = patched
    L = model.num_layers if hasattr(model, "num_layers") else len(model.expert_pools)
    blocks = {"all": range(L), "early": range(0, L // 3), "mid": range(L // 3, 2 * L // 3),
              "late": range(2 * L // 3, L)}
    conds = [("none", None, "all")] + [(f"ko{e}_{b}", e, b) for e in range(model.num_experts)
                                       for b in blocks]
    res = {"run": run, "num_layers": L, "blocks": {b: list(r) for b, r in blocks.items()},
           "logged_final": logged["mean_task_answer_loss"], "conditions": {}}
    for name, e, b in conds:
        state["expert"], state["layers"] = e, set(blocks[b])
        t0 = time.time()
        m = trainer.evaluate("final", None)
        res["conditions"][name] = {"expert": e, "block": b,
                                   "mean_task_answer_loss": m["mean_task_answer_loss"],
                                   "per_task": {t: v["answer_loss"] for t, v in m["per_task"].items()}}
        logger.info(f"{run} {name}: {m['mean_task_answer_loss']:.4f} ({time.time() - t0:.0f}s)")
        if name == "none":
            d = abs(m["mean_task_answer_loss"] - logged["mean_task_answer_loss"])
            logger.info(f"{run}: intact {m['mean_task_answer_loss']:.4f} vs logged "
                        f"{logged['mean_task_answer_loss']:.4f} (|d| {d:.1e})")
            if d > 2e-3:
                raise RuntimeError(f"{run}: intact model does not reproduce the logged test loss")
    out_file.write_text(json.dumps(res, indent=1))
    logger.info(f"{run}: saved {out_file}")
    del trainer, model
    torch.cuda.empty_cache()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="ko_results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "disabled")
    for run in args.runs:
        run_knockout(run, out)


if __name__ == "__main__":
    main()
