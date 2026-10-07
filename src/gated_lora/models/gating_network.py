"""
Gating Network for Expert Selection.

Supports multiple configurations:
- Global gating (one network for all layers)
- Per-layer gating (separate network per transformer layer)
- Dense routing (softmax over all experts)
- Sparse routing (top-k selection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


def masked_token_mean(values: torch.Tensor, token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Mean over the (batch, seq) dims, restricted to real tokens.

    Args:
        values: [batch, seq_len] or [batch, seq_len, k]
        token_mask: [batch, seq_len] (1 = real token, 0 = padding) or None (all tokens)

    Returns:
        Scalar (for [batch, seq_len]) or [k]. No host sync (denominator stays on device).
    """
    if token_mask is None:
        return values.mean(dim=(0, 1))
    m = token_mask.to(device=values.device, dtype=values.dtype)
    if values.dim() == 3:
        m = m.unsqueeze(-1)
    denom = m.sum(dim=(0, 1)).clamp_min(1.0)
    return (values * m).sum(dim=(0, 1)) / denom


class GatingMLP(nn.Module):
    """
    Simple MLP for computing gating logits.

    Architecture: input (+ layer_embedding) -> Linear -> GELU -> Dropout -> Linear -> output

    Per the original Gated LoRA specification, the gating network receives
    both the token embedding AND the layer index to enable layer-specific routing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_layer_embedding: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.use_layer_embedding = use_layer_embedding

        # Layer embedding: learnable embedding for each layer index
        # This allows the gating to differentiate behavior based on layer depth
        if use_layer_embedding and num_layers > 1:
            self.layer_embedding = nn.Embedding(num_layers, input_dim)
            logger.info(f"GatingMLP: Using layer embedding (num_layers={num_layers})")
        else:
            self.layer_embedding = None

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable training."""
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize layer embedding with small values
        if self.layer_embedding is not None:
            nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute gating logits.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            layer_idx: Layer index for layer-aware routing (optional)

        Returns:
            Logits [batch, seq_len, num_experts]
        """
        # Add layer embedding if available and layer_idx provided.
        # Index the weight with the python int directly: no host->device copy
        # (the old `torch.tensor(layer_idx, device=...)` synced once per layer).
        if self.layer_embedding is not None and layer_idx is not None:
            layer_emb = self.layer_embedding.weight[layer_idx]  # [input_dim]
            x = x + layer_emb  # Broadcast to [batch, seq, input_dim]

        return self.gate(x)


class LayerGatingNetwork(nn.Module):
    """
    Per-layer gating network.

    Each transformer layer has its own gating MLP, allowing different
    layers to learn different expert preferences.

    A layer embedding is NOT used here: each gate only ever sees its own layer
    index, so x + e_l followed by Linear(W, b) is just a bias shift (b + W e_l) —
    no extra expressivity, only extra parameters. ``use_layer_embedding`` is
    accepted for signature compatibility and ignored (see GatingNetwork).
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float = 0.1,
        use_layer_embedding: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_experts = num_experts
        # Layer embedding is never allocated with per-layer gates (pure bias shift).
        self.use_layer_embedding = False

        # Create one gating MLP per layer (no layer embedding, see class docstring)
        self.layer_gates = nn.ModuleList([
            GatingMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                dropout=dropout,
                num_layers=num_layers,
                use_layer_embedding=False,
            )
            for _ in range(num_layers)
        ])

        logger.info(f"Created per-layer gating with {num_layers} layers, {num_experts} experts (no layer embedding)")

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute gating logits for a specific layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            layer_idx: Which layer (0-indexed)

        Returns:
            Logits [batch, seq_len, num_experts]
        """
        return self.layer_gates[layer_idx](hidden_states, layer_idx=layer_idx)

    def get_all_layer_logits(
        self,
        hidden_states_per_layer: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gating logits for all layers at once.

        Args:
            hidden_states_per_layer: List of [batch, seq_len, hidden] per layer

        Returns:
            Logits [num_layers, batch, seq_len, num_experts]
        """
        all_logits = []
        for layer_idx, hidden_states in enumerate(hidden_states_per_layer):
            logits = self.forward(hidden_states, layer_idx)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=0)


class GlobalGatingNetwork(nn.Module):
    """
    Global gating network (shared across all layers).

    Simpler and fewer parameters, but less expressive.
    With layer embedding, can still differentiate behavior by layer depth.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_layer_embedding: bool = True,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_layers = num_layers

        self.gate = GatingMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
            num_layers=num_layers,
            use_layer_embedding=use_layer_embedding,
        )

        logger.info(f"Created global gating with {num_experts} experts, layer_embedding={use_layer_embedding}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute gating logits.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            layer_idx: Layer index for layer-aware routing (used with layer embedding)

        Returns:
            Logits [batch, seq_len, num_experts]
        """
        return self.gate(hidden_states, layer_idx=layer_idx)


class GatingNetwork(nn.Module):
    """
    Main gating network that combines:
    - Gating MLP (global or per-layer)
    - Routing strategy (dense softmax or sparse top-k)
    - Load balancing loss computation
    - Routing statistics tracking
    - Layer embedding for layer-aware routing (per Gated LoRA spec)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_layers: int = 1,
        gating_hidden_dim: int = 256,
        gating_dropout: float = 0.1,
        per_layer_gating: bool = True,
        use_top_k: bool = False,
        top_k: int = 2,
        temperature: float = 1.0,
        use_layer_embedding: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim  # Input hidden dim (model's hidden size)
        self.gating_hidden_dim = gating_hidden_dim  # Gating MLP hidden dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.per_layer_gating = per_layer_gating
        self.use_top_k = use_top_k
        self.top_k = min(top_k, num_experts)
        self.temperature = temperature

        uses_per_layer_gates = per_layer_gating and num_layers > 1
        if uses_per_layer_gates and use_layer_embedding:
            logger.warning(
                "use_layer_embedding=True is ignored with per_layer_gating=True: each per-layer "
                "gate only sees its own layer index, so the embedding is a constant bias shift "
                "of the first gate Linear. No layer embedding is allocated."
            )
            use_layer_embedding = False
        # Effective value (False with per-layer gates)
        self.use_layer_embedding = use_layer_embedding

        if use_top_k and self.top_k == 1:
            logger.warning(
                "top_k=1: the softmax over a single selected expert is constant 1.0, so the gate "
                "receives NO gradient from the LM loss (only routing *selection* changes, which "
                "is not differentiable). Use top_k >= 2 or dense routing to train the gate."
            )

        # Create appropriate gating network
        if uses_per_layer_gates:
            self.gating = LayerGatingNetwork(
                num_layers=num_layers,
                input_dim=hidden_dim,
                hidden_dim=gating_hidden_dim,
                num_experts=num_experts,
                dropout=gating_dropout,
                use_layer_embedding=use_layer_embedding,
            )
        else:
            self.gating = GlobalGatingNetwork(
                input_dim=hidden_dim,
                hidden_dim=gating_hidden_dim,
                num_experts=num_experts,
                dropout=gating_dropout,
                num_layers=num_layers,
                use_layer_embedding=use_layer_embedding,
            )

        logger.info(f"GatingNetwork: per_layer={per_layer_gating}, top_k={use_top_k}({top_k}), temp={temperature}, layer_emb={use_layer_embedding}")

    def compute_gate_weights(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
        token_mask: Optional[torch.Tensor] = None,
        compute_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute gating weights (once per layer) and, optionally, routing statistics.

        Temperature is applied before both the dense softmax and the top-k softmax; the
        returned (tempered, possibly sparse) gate weights are what the experts are mixed
        with, for every target module of the layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            layer_idx: Which layer (for per-layer gating / layer embedding)
            token_mask: [batch, seq_len] real-token mask for the statistics (None = all)
            compute_stats: If False, skip the statistics (training fast path); the returned
                routing_info then only holds gate_weights / gate_logits references.

        Returns:
            Tuple of:
                - gate_weights: [batch, seq_len, num_experts] (probabilities; zeros for
                  unselected experts with top-k)
                - gate_logits: [batch, seq_len, num_experts] (raw logits, no temperature)
                - routing_info: Dict with statistics
        """
        # Get raw logits
        gate_logits = self.gating(hidden_states, layer_idx)

        # Apply temperature
        scaled_logits = gate_logits / self.temperature

        if self.use_top_k:
            # Sparse top-k routing
            gate_weights, top_k_indices = self._compute_top_k_weights(scaled_logits)
        else:
            # Dense softmax routing
            gate_weights = F.softmax(scaled_logits, dim=-1)
            top_k_indices = None

        if compute_stats:
            routing_info = self._compute_routing_stats(gate_weights, gate_logits, token_mask)
            if top_k_indices is not None:
                routing_info["top_k_indices"] = top_k_indices
                routing_info["sparsity"] = 1.0 - (self.top_k / self.num_experts)
        else:
            routing_info = {"gate_weights": gate_weights, "gate_logits": gate_logits}

        return gate_weights, gate_logits, routing_info

    def _compute_top_k_weights(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse top-k gating weights.

        Args:
            logits: [batch, seq_len, num_experts] (already divided by the temperature)

        Returns:
            gate_weights: [batch, seq_len, num_experts] (sparse)
            top_k_indices: [batch, seq_len, top_k]
        """
        # Get top-k
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax only on top-k
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Scatter back to full size. Allocate in the softmax dtype: under autocast the
        # logits may be fp16 while softmax returns fp32 (scatter needs matching dtypes).
        gate_weights = torch.zeros(logits.shape, device=logits.device, dtype=top_k_weights.dtype)
        gate_weights = gate_weights.scatter(-1, top_k_indices, top_k_weights)

        return gate_weights, top_k_indices

    def _compute_routing_stats(
        self,
        gate_weights: torch.Tensor,
        gate_logits: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute routing statistics for logging and analysis (real tokens only).

        All values stay on device (no host sync); convert with .item() at logging time.

        Args:
            gate_weights: [batch, seq_len, num_experts]
            gate_logits: [batch, seq_len, num_experts]
            token_mask: [batch, seq_len] (1 = real token) or None (all tokens)

        Returns:
            Dict with various routing metrics
        """
        gate_weights_stats = gate_weights.detach()

        # Expert usage (mean probability per expert over real tokens)
        expert_usage = masked_token_mean(gate_weights_stats, token_mask)  # [num_experts]

        # Entropy of routing distribution (higher = more uncertain)
        eps = 1e-8
        entropy = -torch.sum(gate_weights_stats * torch.log(gate_weights_stats + eps), dim=-1)
        mean_entropy = masked_token_mean(entropy, token_mask)

        # Max entropy for normalization
        max_entropy = math.log(self.num_experts)
        normalized_entropy = mean_entropy / max_entropy

        # Load imbalance (std of expert usage, lower = more balanced)
        load_imbalance = expert_usage.std()

        # Top-1 dominance (how often does one expert dominate)
        top1_probs = gate_weights_stats.max(dim=-1).values
        top1_dominance = masked_token_mean(top1_probs, token_mask)

        return {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "expert_usage": expert_usage,
            "entropy": mean_entropy,
            "normalized_entropy": normalized_entropy,
            "load_imbalance": load_imbalance,
            "top1_dominance": top1_dominance,
        }

    def compute_load_balancing_loss(
        self,
        gate_weights: torch.Tensor,
        gate_logits: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage expert diversity.

        Uses the auxiliary loss from Switch Transformer:
        L_aux = num_experts * sum_i(f_i * P_i)

        where:
        - f_i = fraction of tokens routed to expert i
        - P_i = fraction of router probability allocated to expert i

        Both f and P are averaged over real tokens only (padding excluded).

        Args:
            gate_weights: [batch, seq_len, num_experts]
            gate_logits: [batch, seq_len, num_experts]
            token_mask: [batch, seq_len] (1 = real token) or None (all tokens)

        Returns:
            Scalar loss value
        """
        # f_i: fraction of real tokens where expert i has highest weight (no gradient)
        expert_mask = F.one_hot(gate_weights.argmax(dim=-1), self.num_experts)
        f = masked_token_mean(expert_mask.to(gate_weights.dtype), token_mask)  # [num_experts]

        # P_i: mean probability for each expert over real tokens
        P = masked_token_mean(gate_weights, token_mask)  # [num_experts]

        # Auxiliary loss
        aux_loss = self.num_experts * torch.sum(f * P)

        return aux_loss

    def compute_gate_entropy(
        self,
        gate_weights: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mean gate entropy over real tokens (the historical "L1" gate regulariser).

        `gate_weights.abs().mean()` was a no-op post-softmax (constant 1/num_experts); the
        entropy is the differentiable sparsity surrogate (minimising it sharpens routing).

        Args:
            gate_weights: [batch, seq_len, num_experts]
            token_mask: [batch, seq_len] (1 = real token) or None (all tokens)

        Returns:
            Scalar entropy (nats)
        """
        eps = 1e-10
        entropy = -(gate_weights * (gate_weights + eps).log()).sum(dim=-1)  # [batch, seq]
        return masked_token_mean(entropy, token_mask)

    def compute_entropy_regularization(
        self,
        gate_weights: torch.Tensor,
        target_entropy: float = 0.5,
    ) -> torch.Tensor:
        """
        Regularization to encourage a specific entropy level.

        - Low target_entropy: encourage peaky (specialized) routing
        - High target_entropy: encourage uniform routing

        Args:
            gate_weights: [batch, seq_len, num_experts]
            target_entropy: Target normalized entropy (0-1)

        Returns:
            Scalar loss value
        """
        eps = 1e-8
        max_entropy = math.log(self.num_experts)

        # Current entropy
        entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1)
        normalized_entropy = entropy / max_entropy

        # MSE to target
        return F.mse_loss(normalized_entropy.mean(), torch.tensor(target_entropy, device=gate_weights.device))

    def freeze(self):
        """Freeze gating network parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze gating network parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def num_parameters(self) -> int:
        """Count parameters in gating network."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test the gating network
    print("Testing Gating Network...")

    batch_size = 2
    seq_len = 128
    hidden_dim = 2560
    num_experts = 3
    num_layers = 32

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Test global gating
    print("\n--- Global Gating ---")
    global_gate = GatingNetwork(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_layers=1,
        per_layer_gating=False,
    )
    weights, logits, info = global_gate.compute_gate_weights(x)
    print(f"Weights shape: {weights.shape}")
    print(f"Expert usage: {[f'{v:.3f}' for v in info['expert_usage'].tolist()]}")
    print(f"Entropy: {info['entropy']:.4f}")
    print(f"Params: {global_gate.num_parameters():,}")

    # Test per-layer gating
    print("\n--- Per-Layer Gating ---")
    layer_gate = GatingNetwork(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_layers=num_layers,
        per_layer_gating=True,
    )
    weights, logits, info = layer_gate.compute_gate_weights(x, layer_idx=15)
    print(f"Weights shape: {weights.shape}")
    print(f"Expert usage: {[f'{v:.3f}' for v in info['expert_usage'].tolist()]}")
    print(f"Params: {layer_gate.num_parameters():,}")

    # Test top-k routing
    print("\n--- Top-K Routing ---")
    topk_gate = GatingNetwork(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_layers=1,
        per_layer_gating=False,
        use_top_k=True,
        top_k=2,
    )
    weights, logits, info = topk_gate.compute_gate_weights(x)
    print(f"Weights shape: {weights.shape}")
    print(f"Sparsity: {info['sparsity']:.2f}")
    print(f"Non-zero per token: {(weights > 0).sum(dim=-1).float().mean():.1f}")

    # Test load balancing loss
    print("\n--- Load Balancing Loss ---")
    lb_loss = global_gate.compute_load_balancing_loss(weights, logits)
    print(f"Load balancing loss: {lb_loss:.6f}")

    print("\nAll tests passed!")
