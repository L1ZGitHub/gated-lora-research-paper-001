#!/usr/bin/env python3
"""Compute the trainable-parameter budget of the gated model and the plain-LoRA
rank that matches it — WITHOUT loading base model weights (config.json only).

Instantiates the real LoRAExpertPool / GatingNetwork classes so the count is
exact by construction (no analytic drift if the architecture evolves).

Usage (Ensimag frontale, repo venv):
    .venv/bin/python scripts/analysis/param_match.py --model microsoft/phi-2
    .venv/bin/python scripts/analysis/param_match.py --model microsoft/phi-2 \
        --expert-ranks 8 16 32 --target-modules q_proj k_proj v_proj dense fc1 fc2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformers import AutoConfig  # noqa: E402

from gated_lora.models.gating_network import GatingNetwork  # noqa: E402
from gated_lora.models.lora_experts import LoRAExpertPool  # noqa: E402


def build_pool(cfg, ranks, alphas, target_modules) -> LoRAExpertPool:
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and num_heads:
        head_dim = cfg.hidden_size // num_heads
    return LoRAExpertPool(
        hidden_size=cfg.hidden_size,
        expert_ranks=ranks,
        expert_alphas=alphas,
        target_modules=target_modules,
        dropout=0.0,
        intermediate_size=getattr(cfg, "intermediate_size", cfg.hidden_size * 4),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="microsoft/phi-2")
    p.add_argument("--expert-ranks", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"])
    p.add_argument("--gating-hidden-dim", type=int, default=256)
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    L = cfg.num_hidden_layers
    ranks = args.expert_ranks
    alphas = [2 * r for r in ranks]

    # Gated budget: experts (per-layer pools) + gating network
    pool = build_pool(cfg, ranks, alphas, args.target_modules)
    expert_total = pool.num_parameters() * L

    gating = GatingNetwork(
        hidden_dim=cfg.hidden_size,
        num_experts=len(ranks),
        num_layers=L,
        gating_hidden_dim=args.gating_hidden_dim,
        gating_dropout=0.0,
        per_layer_gating=True,
        use_top_k=False,
        top_k=2,
        temperature=1.0,
        use_layer_embedding=True,
    )
    gating_total = gating.num_parameters()
    gated_total = expert_total + gating_total

    # Plain LoRA: params(r) = r × K, with K = per-layer cost of rank 1 × L
    unit_pool = build_pool(cfg, [1], [2], args.target_modules)
    K = unit_pool.num_parameters() * L

    r_total = gated_total / K
    r_expert = expert_total / K

    print(f"Model: {args.model}  (hidden={cfg.hidden_size}, layers={L})")
    print(f"Target modules: {args.target_modules}")
    print(f"Expert ranks: {ranks}\n")
    print(f"  Expert pools total   : {expert_total:>12,}")
    print(f"  Gating network total : {gating_total:>12,}")
    print(f"  GATED trainable total: {gated_total:>12,}\n")
    print(f"  Plain-LoRA params per rank unit: {K:,}")
    print(f"  → TOTAL-matched rank : r = {r_total:.2f}  → use r={round(r_total)} "
          f"({round(r_total) * K:,} params, "
          f"Δ={100 * (round(r_total) * K - gated_total) / gated_total:+.2f}%)")
    print(f"  → EXPERT-matched rank: r = {r_expert:.2f}  → use r={round(r_expert)} "
          f"({round(r_expert) * K:,} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
