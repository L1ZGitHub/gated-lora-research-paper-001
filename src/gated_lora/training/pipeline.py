"""End-to-end pipeline: build model + dataloaders + trainer, run training.

Ported from legacy ensicompute_harder_multitask/train_v2.py with light cleanup
(no longer assumes Phi-2; trusts the model factory to handle architecture
detection). The pipeline is exposed via two functions:

- ``build_model(config)`` → ``(model, tokenizer)``
- ``build_dataloaders(config, tokenizer)`` → ``(train_loader, eval_loader)``
- ``run_experiment(config)`` → results dict

These match what the CLI consumes in ``gated_lora.cli``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gated_lora.analysis.routing_analysis import analyze_model_routing
from gated_lora.data.multi_task_dataset import MultiTaskDatasetLoader
from gated_lora.models.gated_lora_v2 import create_gated_lora_model
from gated_lora.training.config import ExperimentConfig
from gated_lora.training.gated_trainer import (
    GatedLoRATrainer,
    create_optimizer_and_scheduler,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


def build_baseline_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    mc = config.model
    logger.info("Building baseline LoRA model...")
    tokenizer = AutoTokenizer.from_pretrained(mc.model_name, trust_remote_code=mc.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, mc.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        mc.model_name,
        torch_dtype=torch_dtype,
        device_map=mc.device_map,
        trust_remote_code=mc.trust_remote_code,
    )
    if mc.freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    peft_cfg = LoraConfig(
        r=mc.lora_r,
        lora_alpha=mc.lora_alpha,
        target_modules=mc.lora_target_modules,
        lora_dropout=mc.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.4f}%)")

    return model, tokenizer


def build_gated_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    mc = config.model
    logger.info("Building Gated LoRA v2 model...")
    model = create_gated_lora_model(
        model_name=mc.model_name,
        expert_ranks=mc.expert_ranks,
        expert_alphas=mc.expert_alphas,
        target_modules=mc.lora_target_modules,
        lora_dropout=mc.lora_dropout,
        gating_hidden_dim=mc.gating_hidden_dim,
        gating_dropout=mc.gating_dropout,
        per_layer_gating=mc.per_layer_gating,
        use_top_k=mc.use_top_k,
        top_k=mc.top_k,
        gating_temperature=mc.gating_temperature,
        use_layer_embedding=getattr(mc, "use_layer_embedding", True),
        gated_layers=getattr(mc, "gated_layers", None),
        use_load_balancing=mc.use_load_balancing,
        load_balancing_weight=mc.load_balancing_weight,
        use_l1_gate_regularization=getattr(mc, "use_l1_gate_regularization", True),
        l1_gate_weight=getattr(mc, "l1_gate_weight", 0.01),
        torch_dtype=mc.torch_dtype,
        device_map=mc.device_map,
        trust_remote_code=mc.trust_remote_code,
    )
    return model, model.tokenizer


def build_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    if config.model.model_type == "baseline":
        return build_baseline_model(config)
    return build_gated_model(config)


def build_dataloaders(config: ExperimentConfig, tokenizer) -> Tuple[Any, Any]:
    dc = config.data
    if not dc.use_multi_task:
        raise NotImplementedError(
            "Single-task fallback removed in the unified pipeline. "
            "Provide a tasks: [...] list in your YAML data block."
        )
    logger.info(f"Building multi-task dataloaders ({len(dc.task_datasets)} tasks)...")
    loader = MultiTaskDatasetLoader(
        tokenizer=tokenizer,
        max_length=config.training.max_length,
        task_datasets=dc.task_datasets,
        task_weights=dc.task_weights,
        max_samples_per_task=dc.max_train_samples,
        seed=dc.shuffle_seed,
    )
    train_dl = loader.create_weighted_dataloader(
        split="train",
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory,
    )
    eval_dl = loader.create_eval_dataloader(
        split="validation",
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_num_workers,
    )
    return train_dl, eval_dl


def run_experiment(config: ExperimentConfig, *, analyze_routing: bool = False) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info(f"Running experiment: {config.experiment_name}")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = build_model(config)
    train_dl, eval_dl = build_dataloaders(config, tokenizer)

    total_steps = (
        len(train_dl) * config.training.num_epochs // config.training.gradient_accumulation_steps
    )
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model, config=config, num_training_steps=total_steps
    )

    # wandb is best-effort. Env var WANDB_MODE wins over config.wandb.mode so
    # SLURM jobs can disable it without touching YAML. Init failures (no API
    # key, no network, etc.) must NOT crash training — we already push every
    # artifact to HF Hub.
    env_mode = os.environ.get("WANDB_MODE", "").lower()
    cfg_mode = (config.wandb.mode or "").lower()
    effective_mode = env_mode or cfg_mode
    wandb_off = (
        effective_mode in {"disabled", "off"}
        or not config.wandb.enabled
        or not WANDB_AVAILABLE
    )
    if not wandb_off:
        try:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name or config.experiment_name,
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                config=config.to_dict(),
                mode=effective_mode or "offline",
            )
        except Exception as exc:
            logger.warning(f"wandb.init failed, continuing without wandb: {exc}")

    trainer = GatedLoRATrainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=config.output_dir,
        device=device,
    )
    if config.model.model_type == "gated":
        trainer.set_analysis_dataloader(eval_dl)

    results = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    if config.model.model_type == "gated" and analyze_routing:
        logger.info("Running post-training routing analysis...")
        analysis = analyze_model_routing(
            model=model,
            dataloader=eval_dl,
            tokenizer=tokenizer,
            num_batches=20,
            output_dir=str(Path(config.output_dir) / "visualizations"),
            experiment_name=config.experiment_name,
        )
        results["routing_analysis"] = analysis

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(results)
        wandb.finish()

    out_path = Path(config.output_dir) / "final_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {out_path}")

    return results
