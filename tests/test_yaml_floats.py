"""Regression test for YAML 1.1/1.2 float-parsing footgun.

Without the YAML 1.2 float resolver, ``learning_rate: 2e-4`` parses as the
string ``"2e-4"`` and silently breaks the optimizer constructor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from gated_lora.training import load_config


def test_scientific_notation_parses_as_float(tmp_path: Path):
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text(
        "training:\n"
        "  learning_rate: 2e-4\n"
        "  weight_decay: 1e-2\n"
        "  warmup_ratio: 0.1\n"
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg["training"]["learning_rate"], float)
    assert cfg["training"]["learning_rate"] == 2e-4
    assert isinstance(cfg["training"]["weight_decay"], float)
    assert cfg["training"]["weight_decay"] == 1e-2


def test_repo_configs_have_numeric_learning_rate():
    """All shipped model configs must have a numeric learning_rate."""
    repo_root = Path(__file__).resolve().parents[1]
    for cfg_path in (repo_root / "configs" / "models").glob("*.yaml"):
        cfg = load_config(cfg_path)
        lr = cfg["training"]["learning_rate"]
        assert isinstance(lr, (int, float)), \
            f"{cfg_path.name}: learning_rate is {type(lr).__name__} ({lr!r})"
