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

import contextlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gated_lora.analysis.routing_analysis import analyze_model_routing
from gated_lora.data.multi_task_dataset import MultiTaskDatasetLoader
from gated_lora.models.gated_lora_v2 import create_gated_lora_model
from gated_lora.training.config import (
    ExperimentConfig,
    precision_to_dtype,
    resolve_precision,
)
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


def _ensure_trainable_fp32(model) -> None:
    """Trainable params must be fp32 (AdamW on fp32, autocast for the forward)."""
    cast = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            cast += 1
    if cast:
        logger.info(f"  Cast {cast} trainable parameter tensor(s) to fp32")


def build_baseline_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    mc = config.model
    precision = resolve_precision(mc.precision)
    logger.info(f"Building baseline LoRA model (precision={precision})...")
    tokenizer = AutoTokenizer.from_pretrained(mc.model_name, trust_remote_code=mc.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mc.model_name,
        torch_dtype=precision_to_dtype(precision),
        device_map=mc.device_map,
        trust_remote_code=mc.trust_remote_code,
    )
    model.config.use_cache = False
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
    # PEFT creates adapters in the base dtype: cast them to fp32 (base stays fp16/bf16).
    _ensure_trainable_fp32(model)
    if precision != "fp32":
        # Under autocast PEFT's fp32 copy of every LoRA input is cast straight back to fp16 by
        # the A matmul: skip it (the gated hooks do the same). Only valid with autocast, which
        # the trainer always uses for fp16/bf16.
        n = 0
        for m in model.modules():
            if hasattr(m, "lora_A") and hasattr(m, "cast_input_dtype_enabled"):
                m.cast_input_dtype_enabled = False
                n += 1
        logger.info(f"  PEFT input fp32 cast disabled on {n} LoRA layers (autocast)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.4f}%)")

    return model, tokenizer


def build_gated_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    mc = config.model
    precision = resolve_precision(mc.precision)
    logger.info(f"Building Gated LoRA v2 model (precision={precision})...")
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
        gated_layers=mc.gated_layers,
        gated_layers_frac=mc.gated_layers_frac,
        use_load_balancing=mc.use_load_balancing,
        load_balancing_weight=mc.load_balancing_weight,
        use_l1_gate_regularization=getattr(mc, "use_l1_gate_regularization", True),
        l1_gate_weight=getattr(mc, "l1_gate_weight", 0.01),
        gate_output_scale=mc.gate_output_scale,
        precision=precision,  # resolved here (TRAIN is the source of truth)
        device_map=mc.device_map,
        trust_remote_code=mc.trust_remote_code,
    )
    _ensure_trainable_fp32(model)
    return model, model.tokenizer


@torch.no_grad()
def freeze_and_calibrate_gates(model: Any, data: MultiTaskDatasetLoader,
                               target_top1: Optional[float], autocast_dtype: Any,
                               n_per_task: int = 8, iters: int = 25) -> Dict[str, Any]:
    """Frozen-random-gate control: freeze all gate params; if ``target_top1`` is set, rescale
    each gate MLP's final Linear (per layer, by bisection on log-scale) so that the mean top-1
    gate weight over real tokens of a balanced val batch equals ``target_top1``.

    Exact per layer at init: LoRA B = 0, so a layer's gate does not change any hidden state.
    """
    from gated_lora.models.gating_network import GatingMLP

    for p in model.gating_network.parameters():
        p.requires_grad_(False)
    info: Dict[str, Any] = {"frozen": True, "target_top1": target_top1}
    if target_top1 is None:
        return info
    mlps = [m for m in model.gating_network.modules() if isinstance(m, GatingMLP)]
    per_layer = len(mlps) == model.num_layers
    finals = [m.gate[-1] for m in mlps]
    base_w = [f.weight.detach().clone() for f in finals]
    batch = next(iter(data.create_balanced_loader(n_per_task, batch_size=8 * n_per_task)))
    ids = batch["input_ids"].to(model.device)
    am = batch["attention_mask"].to(model.device)
    lo = [0.0] * len(finals)  # log10 scale bounds
    hi = [4.0] * len(finals)
    was_training = model.training
    model.eval()

    def measure(log_scales):
        for f, w, ls in zip(finals, base_w, log_scales):
            f.weight.copy_(w * (10.0 ** ls))
        ctx = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
        with ctx:
            out = model(input_ids=ids, attention_mask=am, return_routing_info=True)
        pli = out["routing_info"]["per_layer_info"]
        dom = {int(k): float(v["top1_dominance"]) for k, v in pli.items()
               if not v.get("uniform") and "top1_dominance" in v}
        if per_layer:
            return [dom.get(l, target_top1) for l in range(len(finals))]
        return [sum(dom.values()) / max(len(dom), 1)]

    for _ in range(iters):
        mid = [(a + b) / 2 for a, b in zip(lo, hi)]
        got = measure(mid)
        for i, g in enumerate(got):
            if g < target_top1:
                lo[i] = mid[i]
            else:
                hi[i] = mid[i]
    final = [(a + b) / 2 for a, b in zip(lo, hi)]
    got = measure(final)
    model.train(was_training)
    info.update(log10_scales=final, achieved_top1=got)
    logger.info(f"Frozen gates calibrated to top-1 {target_top1}: achieved "
                f"{min(got):.3f}..{max(got):.3f}, scales 10^[{min(final):.2f}..{max(final):.2f}]")
    return info


def build_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    if config.model.model_type == "baseline":
        return build_baseline_model(config)
    return build_gated_model(config)


def build_data(config: ExperimentConfig, tokenizer) -> MultiTaskDatasetLoader:
    """Build the DATA owner's loader object (contract: docs/v2_contract.md)."""
    dc = config.data
    if not dc.use_multi_task:
        raise NotImplementedError(
            "Single-task fallback removed in the unified pipeline. "
            "Provide a tasks: [...] list in your YAML data block."
        )
    logger.info(f"Building multi-task data ({len(dc.task_datasets)} tasks)...")
    return MultiTaskDatasetLoader(
        tokenizer=tokenizer,
        max_length=config.training.max_length,
        task_datasets=dc.task_datasets,
        task_weights=dc.task_weights,
        max_samples_per_task=dc.max_train_samples,
        seed=config.run_seed,
        strict=True,  # a task that fails to load must abort, not silently shrink the mix
        max_val_samples=dc.max_val_samples,
        val_fraction=dc.val_fraction,
        max_final_samples=dc.max_final_samples,
        answer_only_loss=dc.answer_only_loss,
        length_bucketing=dc.length_bucketing,
        cache_dir=dc.cache_dir,
        data_format=dc.data_format,
        split_seed=dc.split_seed,
    )


def _log_data_stats(data, output_dir: str) -> None:
    """Per-task truncation rate / lengths (DATA: get_task_stats). Never fatal."""
    try:
        stats = {which: data.get_task_stats(which=which) for which in ("train",)}
    except Exception as exc:
        logger.warning(f"Data stats unavailable ({type(exc).__name__}: {exc})")
        return
    for task, st in stats["train"].items():
        logger.info(
            f"  [data/train] {task}: n={st.get('n')} trunc={st.get('truncation_rate', 0):.3f} "
            f"prompt={st.get('mean_prompt_len', 0):.0f} answer={st.get('mean_answer_len', 0):.0f} "
            f"total={st.get('mean_total_len', 0):.0f} max={st.get('max_total_len')} "
            f"dropped={st.get('dropped_too_long')} skipped={st.get('skipped_unusable')}"
        )
    path = Path(output_dir) / "data_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def compute_max_steps(config: ExperimentConfig, data) -> int:
    """Optimizer-step budget: max_steps (v2) or the legacy num_epochs equivalent."""
    tc = config.training
    if tc.max_steps > 0:
        return tc.max_steps
    # Legacy: full accumulation windows over num_epochs (leftovers carry over).
    batches_per_epoch = math.ceil(data.train_epoch_size() / tc.batch_size)
    steps = batches_per_epoch * tc.num_epochs // tc.gradient_accumulation_steps
    logger.info(f"Legacy budget: {batches_per_epoch} batches/epoch x {tc.num_epochs} epochs "
                f"/ {tc.gradient_accumulation_steps} accum = {steps} optimizer steps")
    return max(steps, 1)


def run_experiment(config: ExperimentConfig, *, analyze_routing: bool = False) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info(f"Running experiment: {config.experiment_name}")
    logger.info("=" * 70)

    config.validate()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                    f"(capability {torch.cuda.get_device_capability()})")
    precision = resolve_precision(config.model.precision)
    logger.info(f"Precision: {config.model.precision} -> {precision}")

    model, tokenizer = build_model(config)
    data = build_data(config, tokenizer)
    _log_data_stats(data, config.output_dir)
    if config.model.model_type == "gated" and config.model.freeze_gating:
        # Deterministic from the run seed (gate init) and the fixed val split. On resume the
        # calibrated weights are overwritten by the checkpoint anyway (gates are saved).
        gate_info = freeze_and_calibrate_gates(
            model, data, config.model.frozen_gate_target_top1,
            {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision))
        with open(Path(config.output_dir) / "frozen_gate_calibration.json", "w") as f:
            json.dump(gate_info, f, indent=2)

    max_steps = compute_max_steps(config, data)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model, config=config, num_training_steps=max_steps
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
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=config.output_dir,
        device=device,
        max_steps=max_steps,
        precision=precision,
    )

    try:
        results = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    except BaseException:
        # The next slice may land on another node: publish the last consistent local latest/.
        trainer.push_latest_after_crash()
        raise

    completed = results.get("status") in ("completed", "already_complete")
    if (config.model.model_type == "gated" and analyze_routing
            and results.get("status") == "completed"):
        # Never crash the run here (weights are saved), but keep the error visible.
        if trainer.time_allows(trainer.estimate_routing_analysis_seconds()):
            logger.info("Running post-training routing analysis...")
            try:
                prev_collect = getattr(model, "collect_routing_stats", None)
                with trainer.logits_off():  # routing only: skip the vocabulary projection
                    analysis = analyze_model_routing(
                        model=model,
                        dataloader=trainer.get_analysis_dataloader(),
                        tokenizer=tokenizer,
                        num_batches=None,
                        output_dir=str(Path(config.output_dir) / "visualizations"),
                        experiment_name=config.experiment_name,
                        autocast_dtype=trainer.autocast_dtype,
                    )
                if prev_collect is not None:
                    model.collect_routing_stats = prev_collect
                results["routing_analysis"] = analysis
            except Exception as exc:
                logger.exception("Post-training routing analysis FAILED (run results kept)")
                results["routing_analysis_error"] = f"{type(exc).__name__}: {exc}"
        else:
            logger.warning("Post-training routing analysis skipped: deadline too close")
            results["routing_analysis_error"] = "skipped: deadline"

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(results)
        wandb.finish()

    out_path = Path(config.output_dir) / "final_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {out_path}")

    if results.get("status") == "completed":
        # Blocking: best_model, final_model, root files, then TRAINING_DONE (verified).
        trainer.finalize_push()
    else:
        trainer.shutdown_push()
    if not completed:
        logger.info(f"Run not finished (status={results.get('status')}); resume with --resume auto")

    return results
