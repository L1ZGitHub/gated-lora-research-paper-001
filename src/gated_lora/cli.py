"""Entrypoint: `python -m gated_lora.cli --config configs/experiments/X.yaml --seed 42`.

Loads a YAML experiment config, builds an ExperimentConfig dataclass, sets the
seed, and hands off to the trainer. The actual model/dataset/trainer wiring is
deferred — at this stage the CLI verifies config loading end-to-end and prints
the resolved config so SLURM jobs fail fast on misconfiguration.

Phase I will fill in the model/dataset/trainer wiring once the integration
test fixture is in place.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from gated_lora.training import load_config
from gated_lora.training.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)


logger = logging.getLogger(__name__)


def _filter_kwargs(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields that the dataclass actually defines."""
    valid = set(cls.__dataclass_fields__.keys())
    extra = set(data) - valid
    if extra:
        logger.warning(f"{cls.__name__}: ignoring unknown fields {sorted(extra)}")
    return {k: v for k, v in data.items() if k in valid}


def dict_to_experiment_config(cfg: Dict[str, Any]) -> ExperimentConfig:
    """Build a typed ExperimentConfig from a YAML-loaded dict.

    Drops top-level keys that the dataclass doesn't know about
    (slurm, ablation, output) — the SLURM keys are consumed by sbatch
    scripts, ablation is metadata for traceability, and output.output_dir
    is mapped to ExperimentConfig.output_dir explicitly.
    """
    cfg = dict(cfg)  # don't mutate caller's dict
    output_block = cfg.pop("output", None) or {}
    cfg.pop("slurm", None)
    cfg.pop("ablation", None)  # kept on disk via separate metadata
    cfg.pop("description", None)

    model = ModelConfig(**_filter_kwargs(ModelConfig, cfg.pop("model", {})))
    training = TrainingConfig(**_filter_kwargs(TrainingConfig, cfg.pop("training", {})))

    # Translate paper-readable YAML names → legacy DataConfig field names.
    data_block = cfg.pop("data", {}) or {}
    if "tasks" in data_block:
        data_block["task_datasets"] = data_block.pop("tasks")
        data_block["use_multi_task"] = True
    if "weights" in data_block:
        data_block["task_weights"] = data_block.pop("weights")
    if "max_samples_per_task" in data_block:
        # Train cap; only mirror onto the val cap when the YAML doesn't set
        # max_val_samples explicitly (a 5000/task TRAIN cap must not force a
        # 5000/task EVAL — that's a 2-4h evaluation inside a 4h SLURM job).
        cap = data_block.pop("max_samples_per_task")
        data_block["max_train_samples"] = cap
        data_block.setdefault("max_val_samples", cap)
    if "shuffle" in data_block:
        # Legacy doesn't have a "shuffle" toggle — `shuffle_seed` controls reshuffling.
        data_block.pop("shuffle")

    data = DataConfig(**_filter_kwargs(DataConfig, data_block))
    wandb_cfg = WandbConfig(**_filter_kwargs(WandbConfig, cfg.pop("wandb", {})))

    output_dir = output_block.get("output_dir") or cfg.pop("output_dir", None) or "./outputs"

    return ExperimentConfig(
        model=model,
        training=training,
        data=data,
        wandb=wandb_cfg,
        experiment_name=cfg.pop("experiment_name", "gated-lora-exp"),
        output_dir=str(output_dir),
        seed=cfg.pop("seed", 42),
        resume_from_checkpoint=cfg.pop("resume_from_checkpoint", None),
    )


def set_seed(seed: int) -> None:
    """Make training reproducible across NumPy / PyTorch / CUDA / Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gated_lora.cli",
        description="Run a single Gated LoRA training (one config × one seed).",
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--seed", type=int, required=True, help="Random seed")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config's output.output_dir",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help='"auto" or a checkpoint path. Default: from config.',
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config + print it; do not start training.",
    )
    p.add_argument(
        "--analyze-routing",
        action="store_true",
        help="Run post-training routing analysis (gated models only).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    raw = load_config(args.config)
    config = dict_to_experiment_config(raw)

    config.seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.resume:
        config.resume_from_checkpoint = args.resume

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(Path(config.output_dir) / "experiment_config.json")

    logger.info("Resolved config:")
    logger.info(json.dumps(config.to_dict(), indent=2, default=str))

    set_seed(config.seed)

    if args.dry_run:
        logger.info("--dry-run: stopping before training.")
        return 0

    from gated_lora.training.pipeline import run_experiment

    run_experiment(config, analyze_routing=args.analyze_routing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
