"""Smoke tests for model components: LoRAExpert + GatedLoRAModelV2 imports/structure."""

from __future__ import annotations

import inspect

import pytest

from gated_lora.models import (
    GatedLoRAModelV2,
    LoRAExpert,
)


@pytest.mark.parametrize(
    "name,cfg",
    [
        (
            "phi-2",
            dict(
                hidden_size=2560, intermediate_size=10240,
                num_attention_heads=None, num_key_value_heads=None, head_dim=None,
                modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
                expected={"q_proj": (2560, 2560), "fc1": (2560, 10240), "fc2": (10240, 2560)},
            ),
        ),
        (
            "qwen-gqa",
            dict(
                hidden_size=896, intermediate_size=4864,
                num_attention_heads=14, num_key_value_heads=2, head_dim=64,
                modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                expected={"q_proj": (896, 896), "k_proj": (896, 128), "v_proj": (896, 128), "o_proj": (896, 896)},
            ),
        ),
        (
            "llama32-gqa",
            dict(
                hidden_size=3072, intermediate_size=8192,
                num_attention_heads=24, num_key_value_heads=8, head_dim=128,
                modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                expected={"q_proj": (3072, 3072), "k_proj": (3072, 1024), "v_proj": (3072, 1024)},
            ),
        ),
        (
            "pythia-gpt-neox",
            dict(
                hidden_size=1024, intermediate_size=4096,
                num_attention_heads=16, num_key_value_heads=16, head_dim=64,
                modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
                expected={"query_key_value": (1024, 3072), "dense_h_to_4h": (1024, 4096), "dense_4h_to_h": (4096, 1024)},
            ),
        ),
    ],
)
def test_lora_expert_dimensions_per_architecture(name, cfg):
    """LoRAExpert resolves correct (in_dim, out_dim) for every supported architecture."""
    expected = cfg.pop("expected")
    modules = cfg.pop("modules")
    expert = LoRAExpert(rank=8, alpha=16.0, dropout=0.1, target_modules=modules, **cfg)
    for mod_name, (exp_in, exp_out) in expected.items():
        lora = expert.lora_layers[mod_name]
        assert lora.in_features == exp_in, f"{name}/{mod_name}: in {lora.in_features} != {exp_in}"
        assert lora.out_features == exp_out, f"{name}/{mod_name}: out {lora.out_features} != {exp_out}"


def test_gated_lora_v2_features_present():
    """GatedLoRAModelV2 source must contain all merged features (Phi-2 + GQA + Pythia)."""
    src = inspect.getsource(GatedLoRAModelV2)
    for token in ("num_attention_heads", "num_key_value_heads", "head_dim",
                  "use_layer_embedding", "gated_layers",
                  "gpt_neox", "query_key_value"):
        assert token in src, f"Merged feature missing in GatedLoRAModelV2: {token}"


def test_no_buggy_mean_dominance_key():
    """The per_layer fork bug (info['mean_dominance']) must not have leaked in."""
    src = inspect.getsource(GatedLoRAModelV2)
    assert 'info["mean_dominance"]' not in src
    assert "info['mean_dominance']" not in src
