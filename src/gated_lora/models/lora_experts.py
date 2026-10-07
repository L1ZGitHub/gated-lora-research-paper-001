"""
LoRA Expert Modules - Individual LoRA adapters that can be combined via gating.

Each expert is a separate LoRA adapter with its own rank and alpha.
The gating network decides how to weight each expert's contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import math
import logging
import os

logger = logging.getLogger(__name__)

# Fused expert computation (default). GLR_UNFUSED_EXPERTS=1 restores the original per-expert
# loop; kept only to benchmark / check equivalence against the historical implementation.
FUSED_EXPERTS = os.environ.get("GLR_UNFUSED_EXPERTS", "0") != "1"


class LoRALayer(nn.Module):
    """
    Single LoRA layer: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×d}

    This is a standalone LoRA that can be applied to any linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize A with Kaiming, B with zeros (standard LoRA init)."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA output: scaling * B(A(dropout(x)))

        Args:
            x: Input tensor [batch, seq_len, in_features]

        Returns:
            LoRA delta [batch, seq_len, out_features]
        """
        return self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class LoRAExpert(nn.Module):
    """
    A complete LoRA expert that applies LoRA to multiple target modules.

    For example, if target_modules = ["q_proj", "v_proj"], this expert
    contains separate LoRA layers for each.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: int,
        target_modules: List[str],
        dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        # New: support for GQA models like Gemma-2, Llama-3, Qwen
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules

        # For MLP layers that might have different dimensions
        self.intermediate_size = intermediate_size or hidden_size

        # For GQA models (Gemma-2, Llama-3, Qwen, etc.)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim

        # Create LoRA layers for each target module
        self.lora_layers = nn.ModuleDict()

        for module_name in target_modules:
            # Determine dimensions based on module type
            in_dim, out_dim = self._get_module_dimensions(module_name)

            self.lora_layers[module_name] = LoRALayer(
                in_features=in_dim,
                out_features=out_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

        logger.debug(f"Created LoRA expert r={rank}, alpha={alpha} for {target_modules}")

    def _get_module_dimensions(self, module_name: str) -> Tuple[int, int]:
        """Get input and output dimensions for a module, handling GQA correctly."""
        # For GQA models, compute attention dimensions
        if self.num_attention_heads is not None and self.head_dim is not None:
            q_dim = self.num_attention_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
        else:
            # Fallback: assume no GQA (all dimensions = hidden_size)
            q_dim = self.hidden_size
            kv_dim = self.hidden_size

        if module_name == "q_proj":
            return self.hidden_size, q_dim
        elif module_name in ["k_proj", "v_proj"]:
            return self.hidden_size, kv_dim
        elif module_name == "o_proj":
            return q_dim, self.hidden_size
        elif module_name == "gate_proj" or module_name == "up_proj":
            return self.hidden_size, self.intermediate_size
        elif module_name == "down_proj":
            return self.intermediate_size, self.hidden_size
        elif module_name == "dense":  # Phi-2 uses "dense" for output projection, GPT-NeoX for attn output
            return self.hidden_size, self.hidden_size
        elif module_name == "fc1":  # Phi-2 MLP
            return self.hidden_size, self.hidden_size * 4
        elif module_name == "fc2":  # Phi-2 MLP
            return self.hidden_size * 4, self.hidden_size
        # GPT-NeoX / Pythia specific
        elif module_name == "query_key_value":  # Combined QKV projection
            # Output is 3 * hidden_size (Q, K, V concatenated)
            return self.hidden_size, 3 * self.hidden_size
        elif module_name == "dense_h_to_4h":  # GPT-NeoX MLP up projection
            return self.hidden_size, self.intermediate_size
        elif module_name == "dense_4h_to_h":  # GPT-NeoX MLP down projection
            return self.intermediate_size, self.hidden_size
        else:
            # Default: assume square
            return self.hidden_size, self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
    ) -> torch.Tensor:
        """
        Apply LoRA for a specific module.

        Args:
            hidden_states: Input tensor
            module_name: Which module's LoRA to apply

        Returns:
            LoRA delta for that module
        """
        if module_name not in self.lora_layers:
            raise ValueError(f"Module {module_name} not in expert's target modules")

        return self.lora_layers[module_name](hidden_states)

    def get_lora_delta(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
    ) -> torch.Tensor:
        """Alias for forward - clearer naming."""
        return self.forward(hidden_states, module_name)

    def num_parameters(self) -> int:
        """Count trainable parameters in this expert."""
        return sum(p.numel() for p in self.parameters())


class LoRAExpertPool(nn.Module):
    """
    Pool of multiple LoRA experts with different ranks.

    This manages all experts and provides methods to:
    - Get individual expert outputs
    - Get weighted combination of experts
    - Get all expert outputs for gating
    """

    def __init__(
        self,
        hidden_size: int,
        expert_ranks: List[int],
        expert_alphas: List[int],
        target_modules: List[str],
        dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        # New: support for GQA models like Gemma-2, Llama-3, Qwen
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.expert_ranks = expert_ranks
        self.expert_alphas = expert_alphas
        self.target_modules = target_modules
        self.num_experts = len(expert_ranks)

        assert len(expert_ranks) == len(expert_alphas), \
            f"expert_ranks ({len(expert_ranks)}) must match expert_alphas ({len(expert_alphas)})"

        # Create experts
        self.experts = nn.ModuleList([
            LoRAExpert(
                hidden_size=hidden_size,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                dropout=dropout,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
            for rank, alpha in zip(expert_ranks, expert_alphas)
        ])

        logger.info(f"Created LoRA expert pool with {self.num_experts} experts")
        logger.info(f"  Ranks: {expert_ranks}")
        logger.info(f"  Alphas: {expert_alphas}")
        logger.info(f"  Target modules: {target_modules}")
        if num_attention_heads is not None:
            logger.info(f"  GQA: num_heads={num_attention_heads}, num_kv_heads={num_key_value_heads}, head_dim={head_dim}")

    def get_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
    ) -> torch.Tensor:
        """
        Get outputs from all experts for a module.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden]
            module_name: Which module to compute LoRA for

        Returns:
            Expert outputs [batch, seq_len, num_experts, out_dim]
        """
        expert_outputs = []

        for expert in self.experts:
            output = expert.get_lora_delta(hidden_states, module_name)
            expert_outputs.append(output)

        # Stack along new dimension
        # [batch, seq, hidden] -> [batch, seq, num_experts, hidden]
        return torch.stack(expert_outputs, dim=2)

    def get_weighted_output(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
        gate_weights: torch.Tensor,
        output_scale: float = 1.0,
        expanded_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get weighted combination of expert outputs.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden] (already in the adapter
                dtype, or any dtype under autocast)
            module_name: Which module to compute LoRA for
            gate_weights: Gating weights [batch, seq_len, num_experts] (dense softmax or
                sparse top-k weights, zeros for unselected experts)
            output_scale: Multiplier on the mixed output (``gate_output_scale``). Gate
                weights sum to 1, so with output_scale=1 each expert is effectively scaled
                by ~1/num_experts compared with a plain LoRA; output_scale=num_experts makes
                uniform routing equal to a plain LoRA with the summed ranks.
            expanded_gate: Optional precomputed ``expand_gate_weights(gate_weights,
                output_scale)`` [batch, seq_len, sum(ranks)] (fused path only). The model
                computes it once per layer and reuses it for every target module.

        Returns:
            Weighted sum [batch, seq_len, out_dim]
        """
        if FUSED_EXPERTS:
            if expanded_gate is None:
                expanded_gate = self.expand_gate_weights(gate_weights, output_scale)
            return self.fused_output_from_expanded(hidden_states, module_name, expanded_gate)

        # Legacy per-expert loop (GLR_UNFUSED_EXPERTS=1)
        # Get all expert outputs: [batch, seq, num_experts, out_dim]
        expert_outputs = self.get_expert_outputs(hidden_states, module_name)

        # Expand gate weights for broadcasting: [batch, seq, num_experts, 1]
        gate_weights = gate_weights.unsqueeze(-1)

        # Weighted sum: [batch, seq, out_dim]
        weighted_output = (expert_outputs * gate_weights).sum(dim=2)

        if output_scale != 1.0:
            weighted_output = weighted_output * output_scale
        return weighted_output

    def get_top_k_weighted_output(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
        gate_logits: torch.Tensor,
        top_k: int = 2,
        output_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sparse top-k weighted combination of expert outputs from raw logits.

        NOTE: kept for API compatibility / standalone use only. GatedLoRAModelV2 no longer
        calls it: the tempered top-k weights are computed once per layer by
        ``GatingNetwork`` and passed to ``get_weighted_output`` (zeros for unselected
        experts give the same result).

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden]
            module_name: Which module to compute LoRA for
            gate_logits: Raw gating logits [batch, seq_len, num_experts] (no temperature
                applied here)
            top_k: Number of experts to use
            output_scale: Multiplier on the mixed output (see ``get_weighted_output``)

        Returns:
            Tuple of:
                - Weighted sum [batch, seq_len, out_dim]
                - Gate weights used [batch, seq_len, num_experts]
        """
        batch_size, seq_len, _ = hidden_states.shape
        top_k = min(top_k, self.num_experts)

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, top_k, dim=-1)

        # Softmax only on top-k
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Reconstruct full gate weights: unselected experts get weight 0
        # (softmax dtype: under autocast the logits may be fp16 while softmax returns fp32)
        full_gate_weights = torch.zeros(gate_logits.shape, device=gate_logits.device, dtype=top_k_weights.dtype)
        full_gate_weights.scatter_(-1, top_k_indices, top_k_weights)

        if FUSED_EXPERTS:
            # Same math as the loop below. No `expert_mask.any()` host sync.
            return self._fused_output(hidden_states, module_name, full_gate_weights, output_scale), full_gate_weights

        # Compute outputs only for selected experts
        # This is more complex but more efficient for sparse routing

        # Initialize output
        out_dim = self.experts[0].lora_layers[module_name].out_features
        output = None

        # For each expert, compute contribution where it was selected
        for expert_idx, expert in enumerate(self.experts):
            # Find positions where this expert is in top-k
            # [batch, seq, top_k] == expert_idx -> [batch, seq, top_k]
            expert_mask = (top_k_indices == expert_idx)

            if expert_mask.any():
                # Get the weight for this expert where selected
                expert_weight = (top_k_weights * expert_mask.to(top_k_weights.dtype)).sum(dim=-1, keepdim=True)

                # Compute expert output
                expert_output = expert.get_lora_delta(hidden_states, module_name)

                # Add weighted contribution
                contribution = expert_weight * expert_output
                output = contribution if output is None else output + contribution

        if output is None:
            output = torch.zeros(batch_size, seq_len, out_dim, device=hidden_states.device,
                                 dtype=self.param_dtype)
        if output_scale != 1.0:
            output = output * output_scale

        return output, full_gate_weights

    @property
    def param_dtype(self) -> torch.dtype:
        """dtype of the expert parameters (fp32 in v2)."""
        return self.experts[0].lora_layers[self.target_modules[0]].lora_A.weight.dtype

    def expand_gate_weights(
        self,
        gate_weights: torch.Tensor,
        output_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Expand per-expert gate weights to per-rank-column multipliers.

        Returns ``g[..., expert_of_rank] * (alpha_e / r_e) * output_scale`` with shape
        [batch, seq_len, sum(ranks)]. Every expert has the same rank for all target modules,
        so the result depends only on the layer and is shared by all its target modules.
        Computed in the gate dtype (fp32 softmax output), never in the activation dtype.
        """
        expert_of_rank, scale = self._rank_layout(gate_weights.device, gate_weights.dtype, output_scale)
        return gate_weights.index_select(-1, expert_of_rank) * scale

    def fused_output_from_expanded(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
        expanded_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        sum_e g_e * s_e * B_e A_e x, computed as two matmuls over the concatenated ranks:

            h = x @ [A_1; ...; A_E]^T              [batch, seq, R]   (R = sum of ranks)
            h = h * expanded_gate                  (g repeated over each expert's rank block * s)
            out = h @ [B_1 | ... | B_E]^T           [batch, seq, out_dim]

        Exactly equal to the per-expert weighted sum (linearity). Parameters are not
        changed, so checkpoints stay compatible. One difference in training only: the
        input dropout mask is shared by the experts instead of drawn per expert.

        dtype: without autocast, ``hidden_states`` must already be in the parameter dtype
        (the model hook casts it); h * expanded_gate stays in that dtype. Under autocast
        F.linear runs in the autocast dtype, the product with the fp32 expanded gate is
        promoted to fp32 and the second F.linear autocasts again.
        """
        layers = [expert.lora_layers[module_name] for expert in self.experts]
        A = torch.cat([layer.lora_A.weight for layer in layers], dim=0)
        B = torch.cat([layer.lora_B.weight for layer in layers], dim=1)
        h = F.linear(layers[0].dropout(hidden_states), A)
        h = h * expanded_gate
        return F.linear(h, B)

    def _fused_output(
        self,
        hidden_states: torch.Tensor,
        module_name: str,
        gate_weights: torch.Tensor,
        output_scale: float = 1.0,
    ) -> torch.Tensor:
        """Fused weighted output from per-expert gate weights (see ``fused_output_from_expanded``)."""
        expanded_gate = self.expand_gate_weights(gate_weights, output_scale)
        return self.fused_output_from_expanded(hidden_states, module_name, expanded_gate)

    def _rank_layout(self, device, dtype, output_scale: float = 1.0):
        """
        Index of the expert owning each rank column, and the per-column LoRA scaling
        (alpha_e / r_e) * output_scale. Cached per (device, dtype, output_scale); built once
        (one small host->device copy), so no transfer in the steady-state forward path.
        """
        key = (device, dtype, float(output_scale))
        cache = self.__dict__.setdefault("_rank_layout_cache", {})
        if key not in cache:
            ranks = [expert.rank for expert in self.experts]
            scalings = [expert.alpha / expert.rank for expert in self.experts]
            expert_of_rank = torch.tensor(
                [e for e, r in enumerate(ranks) for _ in range(r)], device=device, dtype=torch.long
            )
            scale = torch.tensor(
                [s * float(output_scale) for s, r in zip(scalings, ranks) for _ in range(r)],
                device=device, dtype=dtype,
            )
            cache[key] = (expert_of_rank, scale)
        return cache[key]

    def num_parameters(self) -> int:
        """Total parameters across all experts."""
        return sum(expert.num_parameters() for expert in self.experts)

    def freeze(self):
        """Freeze all expert parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all expert parameters."""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test the modules
    print("Testing LoRA Expert modules...")

    # Test single LoRA layer
    lora = LoRALayer(in_features=2560, out_features=2560, rank=16, alpha=32)
    x = torch.randn(2, 128, 2560)
    out = lora(x)
    print(f"LoRALayer output shape: {out.shape}")

    # Test expert
    expert = LoRAExpert(
        hidden_size=2560,
        rank=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    out = expert.get_lora_delta(x, "q_proj")
    print(f"LoRAExpert output shape: {out.shape}")
    print(f"LoRAExpert params: {expert.num_parameters():,}")

    # Test expert pool
    pool = LoRAExpertPool(
        hidden_size=2560,
        expert_ranks=[8, 16, 32],
        expert_alphas=[16, 32, 64],
        target_modules=["q_proj", "v_proj"],
    )

    # Test getting all outputs
    expert_outputs = pool.get_expert_outputs(x, "q_proj")
    print(f"Expert pool outputs shape: {expert_outputs.shape}")  # [2, 128, 3, 2560]

    # Test weighted output
    gate_weights = torch.softmax(torch.randn(2, 128, 3), dim=-1)
    weighted = pool.get_weighted_output(x, "q_proj", gate_weights)
    print(f"Weighted output shape: {weighted.shape}")  # [2, 128, 2560]

    # Test top-k
    gate_logits = torch.randn(2, 128, 3)
    topk_out, topk_weights = pool.get_top_k_weighted_output(x, "q_proj", gate_logits, top_k=2)
    print(f"Top-K output shape: {topk_out.shape}")
    print(f"Top-K weights shape: {topk_weights.shape}")

    print(f"\nTotal pool parameters: {pool.num_parameters():,}")
    print("All tests passed!")
