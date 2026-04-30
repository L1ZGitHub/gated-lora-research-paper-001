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
        # Add layer embedding if available and layer_idx provided
        if self.layer_embedding is not None and layer_idx is not None:
            layer_emb = self.layer_embedding(
                torch.tensor(layer_idx, device=x.device)
            )  # [input_dim]
            x = x + layer_emb.unsqueeze(0).unsqueeze(0)  # Broadcast to [batch, seq, input_dim]

        return self.gate(x)


class LayerGatingNetwork(nn.Module):
    """
    Per-layer gating network.

    Each transformer layer has its own gating MLP, allowing different
    layers to learn different expert preferences. Each MLP also receives
    a layer embedding to enable layer-aware routing.
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
        self.use_layer_embedding = use_layer_embedding

        # Create one gating MLP per layer
        # Each MLP has its own weights BUT also receives layer index embedding
        # This provides both per-layer specialization AND explicit layer awareness
        self.layer_gates = nn.ModuleList([
            GatingMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                dropout=dropout,
                num_layers=num_layers,
                use_layer_embedding=use_layer_embedding,
            )
            for _ in range(num_layers)
        ])

        logger.info(f"Created per-layer gating with {num_layers} layers, {num_experts} experts, layer_embedding={use_layer_embedding}")

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
        self.use_layer_embedding = use_layer_embedding

        # Create appropriate gating network
        if per_layer_gating and num_layers > 1:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute gating weights with routing statistics.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            layer_idx: Which layer (for per-layer gating)

        Returns:
            Tuple of:
                - gate_weights: [batch, seq_len, num_experts] (probabilities)
                - gate_logits: [batch, seq_len, num_experts] (raw logits)
                - routing_info: Dict with statistics
        """
        # Get raw logits
        gate_logits = self.gating(hidden_states, layer_idx)

        # Apply temperature
        scaled_logits = gate_logits / self.temperature

        if self.use_top_k:
            # Sparse top-k routing
            gate_weights, routing_info = self._compute_top_k_weights(scaled_logits)
        else:
            # Dense softmax routing
            gate_weights = F.softmax(scaled_logits, dim=-1)
            routing_info = self._compute_routing_stats(gate_weights, gate_logits)

        return gate_weights, gate_logits, routing_info

    def _compute_top_k_weights(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute sparse top-k gating weights.

        Args:
            logits: [batch, seq_len, num_experts]

        Returns:
            gate_weights: [batch, seq_len, num_experts] (sparse)
            routing_info: Statistics dict
        """
        # Get top-k
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax only on top-k
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Scatter back to full size
        gate_weights = torch.zeros_like(logits)
        gate_weights.scatter_(-1, top_k_indices, top_k_weights)

        # Compute stats
        routing_info = self._compute_routing_stats(gate_weights, logits)
        routing_info["top_k_indices"] = top_k_indices
        routing_info["sparsity"] = 1.0 - (self.top_k / self.num_experts)

        return gate_weights, routing_info

    def _compute_routing_stats(
        self,
        gate_weights: torch.Tensor,
        gate_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute routing statistics for logging and analysis.

        Args:
            gate_weights: [batch, seq_len, num_experts]
            gate_logits: [batch, seq_len, num_experts]

        Returns:
            Dict with various routing metrics
        """
        # Expert usage (mean probability per expert)
        expert_usage = gate_weights.mean(dim=[0, 1])  # [num_experts]

        # Entropy of routing distribution (higher = more uncertain)
        eps = 1e-8
        entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1)
        mean_entropy = entropy.mean()

        # Max entropy for normalization
        max_entropy = math.log(self.num_experts)
        normalized_entropy = mean_entropy / max_entropy

        # Load imbalance (std of expert usage, lower = more balanced)
        load_imbalance = expert_usage.std()

        # Top-1 dominance (how often does one expert dominate)
        top1_probs = gate_weights.max(dim=-1).values
        top1_dominance = top1_probs.mean()

        # Per-expert usage
        expert_usage_dict = {
            f"expert_{i}_usage": expert_usage[i].item()
            for i in range(self.num_experts)
        }

        return {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "expert_usage": expert_usage,
            "entropy": mean_entropy,
            "normalized_entropy": normalized_entropy,
            "load_imbalance": load_imbalance,
            "top1_dominance": top1_dominance,
            **expert_usage_dict,
        }

    def compute_load_balancing_loss(
        self,
        gate_weights: torch.Tensor,
        gate_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage expert diversity.

        Uses the auxiliary loss from Switch Transformer:
        L_aux = num_experts * sum_i(f_i * P_i)

        where:
        - f_i = fraction of tokens routed to expert i
        - P_i = fraction of router probability allocated to expert i

        Args:
            gate_weights: [batch, seq_len, num_experts]
            gate_logits: [batch, seq_len, num_experts]

        Returns:
            Scalar loss value
        """
        # Flatten batch and sequence
        num_tokens = gate_weights.shape[0] * gate_weights.shape[1]

        # f_i: fraction of tokens where expert i has highest weight
        expert_mask = F.one_hot(gate_weights.argmax(dim=-1), self.num_experts)
        f = expert_mask.to(gate_weights.dtype).sum(dim=[0, 1]) / num_tokens  # [num_experts]

        # P_i: mean probability for each expert
        P = gate_weights.mean(dim=[0, 1])  # [num_experts]

        # Auxiliary loss
        aux_loss = self.num_experts * torch.sum(f * P)

        return aux_loss

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
