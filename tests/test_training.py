"""Smoke tests for the training module — imports + structure, not real training."""

from __future__ import annotations

import inspect

from gated_lora.training import (
    DEFAULT_MAX_RUNTIME_SECONDS,
    GatedLoRATrainer,
    RoutingSnapshot,
    TrainingState,
)


def test_slurm_chaining_features_present():
    """Trainer must expose the SLURM-aware checkpointing surface."""
    src = inspect.getsource(GatedLoRATrainer)
    for token in (
        "max_runtime_seconds",
        "find_latest_checkpoint",
        "_mark_training_done",
        "TRAINING_DONE",
    ):
        assert token in src, f"SLURM chaining feature missing: {token}"


def test_routing_analysis_features_present():
    """Optional routing analysis must be wired into the trainer."""
    src = inspect.getsource(GatedLoRATrainer)
    for token in (
        "run_routing_analysis",
        "set_analysis_dataloader",
        "_save_routing_history",
        "routing_history",
    ):
        assert token in src, f"Routing analysis feature missing: {token}"


def test_training_state_has_batch_idx():
    """`batch_idx` is required for SLURM mid-epoch resume — must be on TrainingState."""
    assert "batch_idx" in TrainingState.__dataclass_fields__


def test_routing_snapshot_schema():
    expected = {
        "step",
        "epoch",
        "layer_expert_usage",
        "task_layer_expert_usage",
        "layer_entropy",
        "specialization_scores",
    }
    got = set(RoutingSnapshot.__dataclass_fields__.keys())
    assert expected.issubset(got), f"Missing fields: {expected - got}"


def test_default_max_runtime_is_below_slurm_4h():
    # Must leave headroom under the 4h SLURM partition limit.
    assert DEFAULT_MAX_RUNTIME_SECONDS < 4 * 3600
    assert DEFAULT_MAX_RUNTIME_SECONDS >= 3 * 3600
