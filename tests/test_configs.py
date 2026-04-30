"""Test that all YAML configs load and resolve correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from gated_lora.cli import dict_to_experiment_config
from gated_lora.training import load_config


CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _all_yaml_files(subdir: str):
    return sorted((CONFIGS_ROOT / subdir).glob("*.yaml"))


@pytest.mark.parametrize("path", _all_yaml_files("models"))
def test_model_configs_load(path):
    cfg = load_config(path)
    assert "model" in cfg
    assert cfg["model"]["model_name"]
    assert "training" in cfg
    assert cfg["training"]["batch_size"] >= 1


@pytest.mark.parametrize("path", _all_yaml_files("tasks"))
def test_task_configs_load(path):
    cfg = load_config(path)
    assert "data" in cfg
    assert isinstance(cfg["data"]["tasks"], list)
    assert len(cfg["data"]["tasks"]) == len(cfg["data"]["weights"])
    assert abs(sum(cfg["data"]["weights"]) - 1.0) < 1e-3, "task weights should sum to ~1.0"


@pytest.mark.parametrize("path", _all_yaml_files("ablations"))
def test_ablation_configs_load(path):
    cfg = load_config(path)
    assert "ablation" in cfg
    assert "name" in cfg["ablation"]


@pytest.mark.parametrize("path", _all_yaml_files("experiments"))
def test_experiment_configs_resolve_to_typed_config(path):
    """Full pipeline: YAML → dict → ExperimentConfig — must succeed without errors."""
    raw = load_config(path)
    config = dict_to_experiment_config(raw)
    config.seed = 42
    # Quick sanity on the typed view
    assert config.experiment_name
    assert config.model.model_name
    assert config.model.model_type in ("baseline", "gated")
    assert config.training.batch_size >= 1
    assert config.data.use_multi_task is True
    assert len(config.data.task_datasets) == len(config.data.task_weights)


def test_ablation_no_l1_overrides_propagate():
    """The ablation `overrides:` block must take precedence over base configs."""
    cfg = load_config(CONFIGS_ROOT / "experiments" / "phi2_ablation_no_l1.yaml")
    assert cfg["model"]["use_l1_gate_regularization"] is False
    assert cfg["model"]["l1_gate_weight"] == 0.0
    assert cfg["ablation"]["name"] == "no_l1"
