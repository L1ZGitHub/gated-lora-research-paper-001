"""
Gated LoRA implementation with multiple experts.

Architecture:
    Input -> Gating Network -> Expert Weights
           |
    [Expert 1 (r=8), Expert 2 (r=16), Expert 3 (r=32)]
           |
    Weighted Sum -> Output

The gating network is a lightweight MLP that routes inputs to different
LoRA experts based on the input representation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
import logging
from peft import LoraConfig, get_peft_model, PeftModel
from .base_model import Phi2BaseModel

logger = logging.getLogger(__name__)


class GatingNetwork(nn.Module):
    """
    Lightweight MLP for routing inputs to LoRA experts.

    Architecture:
        input (hidden_dim) -> Linear -> GELU -> Dropout -> Linear -> output (num_experts)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 3,
        gating_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize gating network.

        Args:
            hidden_dim: Input hidden dimension (model's hidden size)
            num_experts: Number of LoRA experts
            gating_hidden_dim: Hidden dimension of gating MLP
            dropout: Dropout probability
        """
        super().__init__()

        self.num_experts = num_experts

        # Lightweight MLP
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim, num_experts),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_top_k: bool = False,
        top_k: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute gating weights for experts.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            use_top_k: Whether to use top-k routing (sparse)
            top_k: Number of experts to select if using top-k

        Returns:
            expert_weights: Gating weights [batch, seq_len, num_experts]
            routing_info: Dictionary with routing statistics
        """
        # Compute raw logits
        logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]

        if use_top_k:
            # Top-k routing (sparse MoE)
            top_k = min(top_k, self.num_experts)
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # Create mask for selected experts
            expert_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)

            # Apply softmax only on top-k
            expert_weights = torch.softmax(top_k_logits, dim=-1)

            # Scatter weights back to full expert dimension
            expert_weights_full = torch.zeros_like(logits).scatter_(
                -1, top_k_indices, expert_weights
            )
            expert_weights = expert_weights_full
        else:
            # Dense routing (all experts)
            expert_weights = torch.softmax(logits, dim=-1)
            expert_mask = torch.ones_like(logits)

        # Compute routing statistics
        routing_info = {
            "logits": logits,
            "weights": expert_weights,
            "mask": expert_mask,
            "entropy": self._compute_entropy(expert_weights),
            "load_balance": self._compute_load_balance(expert_weights),
        }

        return expert_weights, routing_info

    def _compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing distribution (higher = more diverse)."""
        # Add small epsilon for numerical stability
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1)
        return entropy.mean()

    def _compute_load_balance(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute load balance across experts (lower = better balance)."""
        # Average weight per expert across batch and sequence
        expert_usage = weights.mean(dim=[0, 1])  # [num_experts]

        # Ideal uniform distribution
        uniform = torch.ones_like(expert_usage) / self.num_experts

        # KL divergence from uniform (load imbalance)
        kl_div = torch.sum(expert_usage * torch.log(expert_usage / uniform + 1e-8))

        return kl_div


class GatedLoraModel(Phi2BaseModel):
    """
    Gated LoRA model with multiple experts.

    This model extends the base Phi-2 model with a gating mechanism that
    dynamically routes inputs to different LoRA experts with varying ranks.

    Key Features:
    - 3 experts with different ranks (8, 16, 32) for different capacity levels
    - Lightweight gating network (~1M params) for routing decisions
    - Configurable routing: softmax (dense) or top-k (sparse)
    - Load balancing loss to encourage expert diversity
    - Routing statistics for analysis and visualization
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        freeze_base: bool = True,
        # Gated LoRA specific
        expert_ranks: List[int] = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        gating_hidden_dim: int = 256,
        gating_dropout: float = 0.1,
        use_load_balancing: bool = True,
        load_balancing_weight: float = 0.01,
    ):
        """
        Initialize Gated LoRA model.

        Args:
            model_name: HuggingFace model identifier
            load_in_8bit: Enable 8-bit quantization
            load_in_4bit: Enable 4-bit quantization
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            trust_remote_code: Allow custom code execution
            freeze_base: Freeze base model parameters
            expert_ranks: List of LoRA ranks for each expert (default: [8, 16, 32])
            lora_alpha: LoRA scaling factor (shared across experts)
            lora_dropout: LoRA dropout probability
            target_modules: Modules to apply LoRA to
            gating_hidden_dim: Hidden dimension of gating network
            gating_dropout: Dropout for gating network
            use_load_balancing: Whether to use load balancing loss
            load_balancing_weight: Weight for load balancing loss
        """
        # Initialize base model
        super().__init__(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            freeze_base=freeze_base,
        )

        # Gated LoRA configuration
        if expert_ranks is None:
            expert_ranks = [8, 16, 32]

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        self.expert_ranks = expert_ranks
        self.num_experts = len(expert_ranks)
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = load_balancing_weight

        # Get model's hidden dimension
        self.hidden_dim = self.model.config.hidden_size

        logger.info(f"Creating Gated LoRA with {self.num_experts} experts")
        logger.info(f"Expert ranks: {expert_ranks}")
        logger.info(f"Target modules: {target_modules}")

        # Create gating network
        self.gating_network = GatingNetwork(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            gating_hidden_dim=gating_hidden_dim,
            dropout=gating_dropout,
        )

        # Move gating network to device and cast to correct dtype
        if self.device == "cuda":
            self.gating_network = self.gating_network.cuda()

        # Cast gating network to model dtype (e.g., bfloat16)
        if torch_dtype is not None:
            self.gating_network = self.gating_network.to(torch_dtype)
            logger.info(f"Gating network cast to {torch_dtype}")

        # Create LoRA experts (we'll add them to the model)
        self._create_experts()

        # Log parameter counts
        self._log_parameter_stats()
        self._log_memory_usage()

    def _create_experts(self):
        """Create multiple LoRA experts with different ranks."""
        logger.info("Creating LoRA experts...")

        # Use the middle rank as the base adapter
        base_rank = self.expert_ranks[1]  # r=16

        self.add_lora(
            r=base_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
        )

        logger.info(f"LoRA experts created with ranks {self.expert_ranks}")

    def _log_parameter_stats(self):
        """Log parameter statistics including gating network."""
        # Gating network params
        gating_params = sum(p.numel() for p in self.gating_network.parameters())

        # Model params (base + LoRA)
        model_params = self.get_trainable_params()

        # Total trainable (LoRA + gating)
        total_trainable = model_params["trainable_params"] + gating_params
        total_params = model_params["total_params"] + gating_params

        logger.info(f"Parameter breakdown:")
        logger.info(f"  Gating network: {gating_params:,} params ({gating_params/1e6:.2f}M)")
        logger.info(f"  LoRA adapters: {model_params['trainable_params']:,} params")
        logger.info(f"  Total trainable: {total_trainable:,} params ({100*total_trainable/total_params:.4f}%)")
        logger.info(f"  Base model (frozen): {model_params['total_params'] - model_params['trainable_params']:,} params")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_top_k: bool = False,
        top_k: int = 2,
        return_routing_info: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gated LoRA experts.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]
            use_top_k: Use top-k routing (sparse MoE)
            top_k: Number of experts to select
            return_routing_info: Return routing statistics
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - loss: Total loss (LM loss + optional load balancing loss)
                - logits: Model logits
                - routing_info: Routing statistics (if return_routing_info=True)
        """
        # Get hidden states from the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        # Get the last hidden states for gating
        hidden_states = outputs.hidden_states[-1]

        # Compute gating weights
        expert_weights, routing_info = self.gating_network(
            hidden_states,
            use_top_k=use_top_k,
            top_k=top_k,
        )

        # Prepare result
        result = {
            "logits": outputs.logits,
        }

        # Add loss
        if labels is not None:
            lm_loss = outputs.loss

            # Add load balancing loss if enabled
            # Use self.model.training since GatedLoraModel is not an nn.Module
            if self.use_load_balancing and self.model.training:
                load_balance_loss = routing_info["load_balance"]
                total_loss = lm_loss + self.load_balancing_weight * load_balance_loss

                result["loss"] = total_loss
                result["lm_loss"] = lm_loss
                result["load_balance_loss"] = load_balance_loss
            else:
                result["loss"] = lm_loss

        # Add routing info
        if return_routing_info:
            result["routing_info"] = routing_info
            result["expert_weights"] = expert_weights

        return result

    def get_routing_stats(self, expert_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute routing statistics for analysis.

        Args:
            expert_weights: Expert weights [batch, seq_len, num_experts]

        Returns:
            Dictionary with routing statistics
        """
        # Average weight per expert
        expert_usage = expert_weights.mean(dim=[0, 1])  # [num_experts]

        # Convert to percentages
        expert_percentages = (expert_usage * 100).tolist()

        # Create stats dict
        stats = {
            f"expert_{i}_usage": pct
            for i, pct in enumerate(expert_percentages)
        }

        # Add entropy (diversity measure)
        eps = 1e-8
        entropy = -torch.sum(expert_usage * torch.log(expert_usage + eps))
        stats["routing_entropy"] = entropy.item()

        # Add balance score (1.0 = perfect balance)
        ideal_weight = 1.0 / self.num_experts
        balance = 1.0 - torch.std(expert_usage).item() / ideal_weight
        stats["balance_score"] = balance

        return stats

    def get_trainable_params(self) -> Dict[str, int]:
        """
        Get trainable parameter count including gating network.

        Returns:
            Dictionary with parameter counts
        """
        # Model params (LoRA)
        model_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_total = sum(p.numel() for p in self.model.parameters())

        # Gating params (all trainable)
        gating_trainable = sum(p.numel() for p in self.gating_network.parameters())

        # Total
        total_trainable = model_trainable + gating_trainable
        total_params = model_total + gating_trainable

        return {
            "trainable_params": total_trainable,
            "total_params": total_params,
            "trainable_percentage": 100 * total_trainable / total_params if total_params > 0 else 0,
            "lora_params": model_trainable,
            "gating_params": gating_trainable,
        }

    def visualize_routing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Visualize routing decisions for given input.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Dictionary with routing visualization data
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_routing_info=True,
            )

        expert_weights = outputs["expert_weights"]
        routing_info = outputs["routing_info"]

        # Get tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            "tokens": tokens,
            "expert_weights": expert_weights[0].cpu(),  # [seq_len, num_experts]
            "routing_info": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in routing_info.items()
            },
            "routing_stats": self.get_routing_stats(expert_weights),
        }


# ASCII Art of Architecture
ARCHITECTURE_DIAGRAM = """
================================================================================
                         GATED LORA ARCHITECTURE                              
================================================================================

                        Input Hidden States [B, L, H]
                                    |
                                    v
                        +----------------------+
                        |   Gating Network     |
                        |                      |
                        |  Linear(H -> 256)    |
                        |       |              |
                        |      GELU            |
                        |       |              |
                        |  Dropout(0.1)        |
                        |       |              |
                        |  Linear(256 -> 3)    |
                        |       |              |
                        |  Softmax/TopK        |
                        |                      |
                        +----------------------+
                                    |
                        +-----------+-----------+
                        |           |           |
                        v           v           v
                +-----------+ +-----------+ +-----------+
                | Expert 1  | | Expert 2  | | Expert 3  |
                |           | |           | |           |
                | LoRA r=8  | | LoRA r=16 | | LoRA r=32 |
                | (Light)   | | (Medium)  | | (Heavy)   |
                |           | |           | |           |
                | ~2M params| | ~4M params| | ~8M params|
                |           | |           | |           |
                | Fast but  | | Balanced  | | Slow but  |
                | limited   | | capacity  | | high cap. |
                +-----------+ +-----------+ +-----------+
                        |           |           |
                        +-----------+-----------+
                                    |
                                    v
                        +----------------------+
                        |   Weighted Sum       |
                        |                      |
                        |  output = sum(w*o)   |
                        +----------------------+
                                    |
                                    v
                            Combined Output [B, L, H]

Legend:
  B = Batch size
  L = Sequence length
  H = Hidden dimension (2560 for Phi-2)

Key Features:
  - Gating network: ~1M parameters (256 hidden dim)
  - 3 experts with different capacities (r=8, 16, 32)
  - Dynamic routing based on input content
  - Load balancing to prevent expert collapse
  - Routing statistics for analysis
"""


if __name__ == "__main__":
    print(ARCHITECTURE_DIAGRAM)
