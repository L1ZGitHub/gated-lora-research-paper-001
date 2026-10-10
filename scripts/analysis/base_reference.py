#!/usr/bin/env python3
"""Base-model loss per task on the full val and final (test) splits (GPU, no training).

The adapters are zero-initialised (LoRA B=0), so an untrained plain-LoRA model IS the base
model. Used for the v3 robustness check: mean over tasks of loss / base-model loss on the
same split (the primary metric is the plain mean task loss; experiments/README.md, v3).

    PYTHONPATH=src .venv/bin/python scripts/analysis/base_reference.py \
        --config configs/experiments/v3/v3_qwen25_05b_baseline_r32_lr5em5.yaml \
        --out experiments/v3_base_reference.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from gated_lora.cli import dict_to_experiment_config
from gated_lora.training import load_config
from gated_lora.training.config import resolve_precision
from gated_lora.training.gated_trainer import GatedLoRATrainer, create_optimizer_and_scheduler
from gated_lora.training.pipeline import build_data, build_model

logger = logging.getLogger("base_reference")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a plain-LoRA (baseline) config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")

    config = dict_to_experiment_config(load_config(args.config))
    if config.model.model_type != "baseline":
        raise SystemExit("use a plain-LoRA config: its untrained model is exactly the base model")
    config.output_dir = tempfile.mkdtemp(prefix="base_ref_")
    config.training.push_to_hub = False
    config.wandb.enabled = False
    config.validate()

    model, tokenizer = build_model(config)
    data = build_data(config, tokenizer)
    opt, sched = create_optimizer_and_scheduler(model=model, config=config, num_training_steps=1)
    trainer = GatedLoRATrainer(model=model, data=data, optimizer=opt, scheduler=sched, config=config,
                               output_dir=config.output_dir, device="cuda", max_steps=1,
                               precision=resolve_precision(config.model.precision))
    res = {"config": args.config, "data_format": config.data.data_format}
    for split in ("val", "final"):
        m = trainer.evaluate(split, None)
        res[split] = {t: v["answer_loss"] for t, v in m["per_task"].items()}
        res[f"{split}_num_examples"] = {t: v["num_examples"] for t, v in m["per_task"].items()}
        logger.info(f"{split}: " + " ".join(f"{t}={v:.4f}" for t, v in res[split].items()))
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    logger.info(f"saved {args.out}")


if __name__ == "__main__":
    main()
