"""
Gated LoRA Model v2 - Complete rewrite with real expert routing.

This implementation follows the original research plan:
- Multiple separate LoRA experts with different ranks
- Per-layer gating network for expert selection
- Token-level routing decisions
- Dense or sparse (top-k) routing
- Load balancing loss for expert diversity

Architecture:
    Input -> Base Model Layer -> Hidden States
                                      |
                                      v
                              Gating Network (per layer)
                                      |
                              [g1, g2, g3] weights
                                      |
            +-------------------------+-------------------------+
            |                         |                         |
            v                         v                         v
      Expert 1 (r=8)           Expert 2 (r=16)          Expert 3 (r=32)
            |                         |                         |
            v                         v                         v
         delta_1                   delta_2                   delta_3
            |                         |                         |
            +-------------------------+-------------------------+
                                      |
                                      v
                    Weighted Sum: g1*delta_1 + g2*delta_2 + g3*delta_3
                                      |
                                      v
                              Base Output + Weighted Sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Callable
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig

from .lora_experts import LoRAExpertPool
from .gating_network import GatingNetwork

logger = logging.getLogger(__name__)


class GatedLoRAModelV2(nn.Module):
    """
    Gated LoRA Model with real expert routing.

    This model wraps a pretrained transformer and adds:
    1. Multiple LoRA experts with different capacities
    2. A gating network that decides expert weights per token
    3. Hooks to intercept and modify layer outputs
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        # Expert configuration
        expert_ranks: List[int] = None,
        expert_alphas: List[int] = None,
        target_modules: List[str] = None,
        lora_dropout: float = 0.1,
        # Gating configuration
        gating_hidden_dim: int = 256,
        gating_dropout: float = 0.1,
        per_layer_gating: bool = True,
        use_top_k: bool = False,
        top_k: int = 2,
        gating_temperature: float = 1.0,
        use_layer_embedding: bool = True,  # NEW: Layer embedding in gating
        gated_layers: List[int] = None,  # NEW: Only apply gating to these layers (None = all)
        # Load balancing
        use_load_balancing: bool = True,
        load_balancing_weight: float = 0.001,
        # L1 regularization on gates (per Gated LoRA spec)
        use_l1_gate_regularization: bool = True,
        l1_gate_weight: float = 0.01,
        # Model loading
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        super().__init__()

        # Defaults
        if expert_ranks is None:
            expert_ranks = [8, 16, 32]
        if expert_alphas is None:
            expert_alphas = [16, 32, 64]
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "dense"]

        self.model_name = model_name
        self.expert_ranks = expert_ranks
        self.expert_alphas = expert_alphas
        self.target_modules = target_modules
        self.num_experts = len(expert_ranks)
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = load_balancing_weight
        self.use_l1_gate_regularization = use_l1_gate_regularization
        self.l1_gate_weight = l1_gate_weight
        self.per_layer_gating = per_layer_gating
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.use_layer_embedding = use_layer_embedding
        self.gated_layers = gated_layers  # None means all layers

        logger.info(f"Initializing GatedLoRAModelV2 with {self.num_experts} experts")
        logger.info(f"  Ranks: {expert_ranks}")
        logger.info(f"  Alphas: {expert_alphas}")
        logger.info(f"  Target modules: {target_modules}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info(f"Loading base model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")

        # Get model dimensions
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.intermediate_size = getattr(self.config, 'intermediate_size', self.hidden_size * 4)

        # Get attention dimensions for GQA models (Gemma-2, Llama-3, Qwen, etc.)
        self.num_attention_heads = getattr(self.config, 'num_attention_heads', None)
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_attention_heads)
        self.head_dim = getattr(self.config, 'head_dim', None)

        # If head_dim not in config, try to compute it
        if self.head_dim is None and self.num_attention_heads is not None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        logger.info(f"Model config: hidden={self.hidden_size}, layers={self.num_layers}")
        if self.num_attention_heads is not None:
            logger.info(f"Attention config: num_heads={self.num_attention_heads}, "
                       f"num_kv_heads={self.num_key_value_heads}, head_dim={self.head_dim}")

        # Create expert pool (per-layer experts)
        self.expert_pools = nn.ModuleList([
            LoRAExpertPool(
                hidden_size=self.hidden_size,
                expert_ranks=expert_ranks,
                expert_alphas=expert_alphas,
                target_modules=target_modules,
                dropout=lora_dropout,
                intermediate_size=self.intermediate_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
            )
            for _ in range(self.num_layers)
        ])

        # Create gating network
        self.gating_network = GatingNetwork(
            hidden_dim=self.hidden_size,
            num_experts=self.num_experts,
            num_layers=self.num_layers,
            gating_hidden_dim=gating_hidden_dim,
            gating_dropout=gating_dropout,
            per_layer_gating=per_layer_gating,
            use_top_k=use_top_k,
            top_k=top_k,
            temperature=gating_temperature,
            use_layer_embedding=use_layer_embedding,  # NEW
        )

        # Log gated layers config
        if self.gated_layers is not None:
            logger.info(f"  Partial gating: only layers {self.gated_layers}")
        else:
            logger.info(f"  Full gating: all {self.num_layers} layers")

        # Move to device and dtype
        self.device = next(self.model.parameters()).device
        self.dtype = torch_dtype

        # Move experts and gating to correct device/dtype
        for pool in self.expert_pools:
            pool.to(device=self.device, dtype=self.dtype)
        self.gating_network.to(device=self.device, dtype=self.dtype)

        # Storage for routing info during forward pass
        self._routing_info_per_layer: Dict[int, Dict] = {}
        self._accumulated_load_balance_loss = 0.0
        self._accumulated_l1_gate_loss = 0.0
        self._gating_cache: Dict[Tuple[int, int], Tuple] = {}  # Cache gating per layer

        # Register hooks for each target module
        self._register_hooks()

        # Log parameter counts
        self._log_parameter_counts()

    def _register_hooks(self):
        """Register forward hooks on each target Linear module individually."""
        self._hooks = []

        # Find the transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama, Mistral, Gemma, etc.
            layers = self.model.model.layers
            logger.info(f"Found transformer layers via model.model.layers (Llama-style)")
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            layers = self.model.transformer.h
            logger.info(f"Found transformer layers via model.transformer.h (GPT-2 style)")
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            # Pythia / GPT-NeoX style
            layers = self.model.gpt_neox.layers
            logger.info(f"Found transformer layers via model.gpt_neox.layers (Pythia/GPT-NeoX style)")
        else:
            logger.error("Could not find transformer layers for hook registration!")
            logger.error(f"Model structure: {type(self.model)}")
            logger.error(f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
            return

        # Register hook for each target module in each layer
        modules_found = 0
        modules_not_found = 0
        for layer_idx, layer in enumerate(layers):
            for module_name in self.target_modules:
                module = self._get_submodule(layer, module_name)
                if module is not None:
                    hook = module.register_forward_hook(
                        self._create_module_hook(layer_idx, module_name)
                    )
                    self._hooks.append(hook)
                    modules_found += 1
                    if layer_idx == 0:  # Log only for first layer to avoid spam
                        logger.info(f"  Found module '{module_name}' in layer 0: {type(module)}")
                else:
                    modules_not_found += 1
                    if layer_idx == 0:  # Log only for first layer
                        logger.warning(f"  Could not find module '{module_name}' in layer 0")
                        # Debug: show layer structure
                        logger.warning(f"  Layer type: {type(layer)}")
                        logger.warning(f"  Layer attrs: {[a for a in dir(layer) if not a.startswith('_')]}")

        logger.info(f"Registered {len(self._hooks)} module hooks "
                   f"({len(self.target_modules)} modules x {self.num_layers} layers)")
        if modules_not_found > 0:
            logger.error(f"WARNING: {modules_not_found} modules not found! "
                        f"Found: {modules_found}, Not found: {modules_not_found}")
            logger.error("This will cause training to fail - LoRA deltas won't be applied!")

    def _get_submodule(self, layer, module_name: str) -> Optional[nn.Module]:
        """Get a submodule from a transformer layer by name."""
        # For Phi-2, Llama, etc.: q_proj, k_proj, v_proj, dense are in self_attn
        if module_name in ["q_proj", "k_proj", "v_proj", "dense", "o_proj"]:
            if hasattr(layer, 'self_attn'):
                return getattr(layer.self_attn, module_name, None)
            elif hasattr(layer, 'attention'):
                return getattr(layer.attention, module_name, None)
        # For GPT-NeoX / Pythia: query_key_value (combined QKV) and dense
        elif module_name in ["query_key_value"]:
            if hasattr(layer, 'attention'):
                return getattr(layer.attention, module_name, None)
        # For MLP modules
        elif module_name in ["fc1", "fc2", "gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer, 'mlp'):
                return getattr(layer.mlp, module_name, None)
        # For GPT-NeoX MLP modules
        elif module_name in ["dense_h_to_4h", "dense_4h_to_h"]:
            if hasattr(layer, 'mlp'):
                return getattr(layer.mlp, module_name, None)
        return None

    def _create_module_hook(self, layer_idx: int, module_name: str) -> Callable:
        """Create a forward hook for a specific module in a layer."""
        def hook(module, inputs, outputs):
            # inputs[0] is the input to the Linear: [batch, seq, in_dim]
            hidden_states = inputs[0]

            # Check if this layer uses gating (for partial gating ablation)
            use_gating_for_layer = (
                self.gated_layers is None or layer_idx in self.gated_layers
            )

            # Use layer_idx as cache key - all modules in a layer share the same gating
            # The gating is computed only once per layer using hidden_size dimension input
            cache_key = layer_idx

            if cache_key not in self._gating_cache:
                # Only compute gating if input dimension matches hidden_size
                # For GQA models, o_proj input has different dimension (num_heads * head_dim)
                # In that case, we must wait for a module with correct input dimension
                input_dim = hidden_states.shape[-1]

                if input_dim != self.hidden_size:
                    # This module (likely o_proj) has wrong input dimension for gating
                    # Skip and let another module compute gating first
                    # This should not happen in normal execution order (q/k/v come before o)
                    raise RuntimeError(
                        f"Gating cache miss for layer {layer_idx}, module {module_name}. "
                        f"Input dim {input_dim} != hidden_size {self.hidden_size}. "
                        f"This may indicate hooks are firing in unexpected order."
                    )

                if use_gating_for_layer:
                    # Compute gating weights normally
                    gate_weights, gate_logits, routing_info = \
                        self.gating_network.compute_gate_weights(hidden_states, layer_idx)
                else:
                    # Uniform routing for non-gated layers
                    batch_size, seq_len = hidden_states.shape[:2]
                    uniform_weight = 1.0 / self.num_experts
                    gate_weights = torch.full(
                        (batch_size, seq_len, self.num_experts),
                        uniform_weight,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    gate_logits = torch.zeros_like(gate_weights)
                    routing_info = {"gate_weights": gate_weights, "uniform": True}

                # Store in cache
                self._gating_cache[cache_key] = (gate_weights, gate_logits, routing_info)

                # Store routing info (once per layer)
                self._routing_info_per_layer[layer_idx] = routing_info

                # Accumulate load balance loss (once per layer, only for gated layers)
                if self.use_load_balancing and self.training and use_gating_for_layer:
                    lb_loss = self.gating_network.compute_load_balancing_loss(
                        gate_weights, gate_logits
                    )
                    self._accumulated_load_balance_loss += lb_loss

                # Accumulate gate sparsity regularization (once per layer, only
                # for gated layers).
                #
                # NOTE (2026-07 fix): the historical formulation
                # `gate_weights.abs().mean()` was a mathematical NO-OP: gate
                # weights come out of a softmax (positive, sum to 1 across the
                # expert dim), so the mean of their absolute values is the
                # constant 1/num_experts — zero gradient, no learning effect.
                # The intent of the "L1" penalty was routing SPARSITY; the
                # correct differentiable surrogate post-softmax is the entropy
                # of the gate distribution (minimizing it sharpens routing).
                # Config keys keep their historical names (use_l1_gate_
                # regularization / l1_gate_weight) so existing YAMLs still work.
                if self.use_l1_gate_regularization and self.training and use_gating_for_layer:
                    eps = 1e-10
                    gate_entropy = -(gate_weights * (gate_weights + eps).log()).sum(dim=-1).mean()
                    self._accumulated_l1_gate_loss += gate_entropy
            else:
                gate_weights, gate_logits, routing_info = self._gating_cache[cache_key]

            # Get expert pool for this layer
            expert_pool = self.expert_pools[layer_idx]

            # Apply LoRA for THIS specific module
            if self.use_top_k:
                lora_delta, final_weights = expert_pool.get_top_k_weighted_output(
                    hidden_states,
                    module_name=module_name,
                    gate_logits=gate_logits,
                    top_k=self.top_k,
                )
                # Update routing info with final weights (only first time)
                if "gate_weights" not in routing_info:
                    routing_info["gate_weights"] = final_weights
            else:
                lora_delta = expert_pool.get_weighted_output(
                    hidden_states,
                    module_name=module_name,
                    gate_weights=gate_weights,
                )

            # Add LoRA delta to the Linear output
            return outputs + lora_delta

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gated LoRA.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] for loss computation
            return_routing_info: Whether to return detailed routing info
            **kwargs: Additional arguments for base model

        Returns:
            Dict with loss, logits, and optionally routing info
        """
        # Reset accumulated loss and caches
        self._accumulated_load_balance_loss = 0.0
        self._accumulated_l1_gate_loss = 0.0
        self._routing_info_per_layer = {}
        self._gating_cache = {}  # Reset gating cache for new forward pass

        # Forward through base model (hooks will apply LoRA)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # Build result
        result = {
            "logits": outputs.logits,
        }

        if labels is not None:
            lm_loss = outputs.loss
            total_loss = lm_loss

            # Add load balancing loss
            if self.use_load_balancing and self.training:
                total_loss = total_loss + self.load_balancing_weight * self._accumulated_load_balance_loss
                result["load_balancing_loss"] = self._accumulated_load_balance_loss

            # Add L1 gate regularization
            if self.use_l1_gate_regularization and self.training:
                total_loss = total_loss + self.l1_gate_weight * self._accumulated_l1_gate_loss
                result["l1_gate_loss"] = self._accumulated_l1_gate_loss

            result["loss"] = total_loss
            result["lm_loss"] = lm_loss

        # Add routing info
        if return_routing_info and self._routing_info_per_layer:
            result["routing_info"] = self._aggregate_routing_info()

        return result

    def _aggregate_routing_info(self) -> Dict[str, Any]:
        """Aggregate routing info across all layers."""
        if not self._routing_info_per_layer:
            return {}

        # Collect per-layer stats (only from gated layers that have entropy)
        all_entropies = []
        all_expert_usages = []
        all_dominances = []

        for layer_idx, info in self._routing_info_per_layer.items():
            # Skip uniform/non-gated layers that don't have entropy stats
            if info.get("uniform", False) or "entropy" not in info:
                continue
            all_entropies.append(info["entropy"])
            all_expert_usages.append(info["expert_usage"])
            all_dominances.append(info["top1_dominance"])

        # Handle case where no gated layers have stats yet
        if not all_entropies:
            return {
                "per_layer_info": self._routing_info_per_layer,
                "num_layers_with_info": len(self._routing_info_per_layer),
                "num_gated_layers": 0,
            }

        # Stack and average
        mean_entropy = torch.stack(all_entropies).mean()
        mean_expert_usage = torch.stack(all_expert_usages).mean(dim=0)
        mean_dominance = torch.stack(all_dominances).mean()

        return {
            "mean_entropy": mean_entropy,
            "mean_expert_usage": mean_expert_usage,
            "mean_top1_dominance": mean_dominance,
            "per_layer_info": self._routing_info_per_layer,
            "num_layers_with_info": len(self._routing_info_per_layer),
            "num_gated_layers": len(all_entropies),
        }

    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics for logging."""
        if not self._routing_info_per_layer:
            return {}

        info = self._aggregate_routing_info()

        # Handle case where no gated layers have stats
        if info.get("num_gated_layers", 0) == 0:
            return {"routing/num_gated_layers": 0}

        stats = {
            "routing/mean_entropy": info["mean_entropy"].item(),
            "routing/mean_top1_dominance": info["mean_top1_dominance"].item() if isinstance(info["mean_top1_dominance"], torch.Tensor) else info["mean_top1_dominance"],
            "routing/num_gated_layers": info.get("num_gated_layers", len(self._routing_info_per_layer)),
        }

        # Add per-expert usage
        for i, usage in enumerate(info["mean_expert_usage"]):
            stats[f"routing/expert_{i}_usage"] = usage.item()

        return stats

    def freeze_experts(self):
        """Freeze all expert parameters (for gating warmup)."""
        for pool in self.expert_pools:
            pool.freeze()
        logger.info("Expert pools frozen")

    def unfreeze_experts(self):
        """Unfreeze all expert parameters."""
        for pool in self.expert_pools:
            pool.unfreeze()
        logger.info("Expert pools unfrozen")

    def freeze_gating(self):
        """Freeze gating network."""
        self.gating_network.freeze()
        logger.info("Gating network frozen")

    def unfreeze_gating(self):
        """Unfreeze gating network."""
        self.gating_network.unfreeze()
        logger.info("Gating network unfrozen")

    def _log_parameter_counts(self):
        """Log parameter counts."""
        # Expert params
        expert_params = sum(pool.num_parameters() for pool in self.expert_pools)

        # Gating params
        gating_params = self.gating_network.num_parameters()

        # Total trainable
        total_trainable = expert_params + gating_params

        # Base model
        base_params = sum(p.numel() for p in self.model.parameters())

        logger.info("Parameter counts:")
        logger.info(f"  Base model (frozen): {base_params:,}")
        logger.info(f"  Expert pools: {expert_params:,} ({expert_params/1e6:.2f}M)")
        logger.info(f"  Gating network: {gating_params:,} ({gating_params/1e6:.2f}M)")
        logger.info(f"  Total trainable: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
        logger.info(f"  Trainable %: {100*total_trainable/(base_params+total_trainable):.4f}%")

    def get_trainable_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        expert_params = sum(pool.num_parameters() for pool in self.expert_pools)
        gating_params = self.gating_network.num_parameters()
        total_trainable = expert_params + gating_params
        base_params = sum(p.numel() for p in self.model.parameters())

        return {
            "expert_params": expert_params,
            "gating_params": gating_params,
            "trainable_params": total_trainable,
            "total_params": base_params + total_trainable,
            "trainable_percentage": 100 * total_trainable / (base_params + total_trainable),
        }

    def save_pretrained(self, save_directory: str):
        """Save model (experts + gating only, not base model)."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save expert pools
        torch.save(
            {f"layer_{i}": pool.state_dict() for i, pool in enumerate(self.expert_pools)},
            os.path.join(save_directory, "expert_pools.pt")
        )

        # Save gating network
        torch.save(
            self.gating_network.state_dict(),
            os.path.join(save_directory, "gating_network.pt")
        )

        # Save config (include ALL parameters needed for reconstruction)
        config = {
            "model_name": self.model_name,
            "expert_ranks": self.expert_ranks,
            "expert_alphas": self.expert_alphas,
            "target_modules": self.target_modules,
            "num_experts": self.num_experts,
            "per_layer_gating": self.per_layer_gating,
            "use_top_k": self.use_top_k,
            "top_k": self.top_k,
            "use_load_balancing": self.use_load_balancing,
            "load_balancing_weight": self.load_balancing_weight,
            "use_l1_gate_regularization": self.use_l1_gate_regularization,
            "l1_gate_weight": self.l1_gate_weight,
            # NEW: Include layer embedding and partial gating config
            "use_layer_embedding": self.use_layer_embedding,
            "gated_layers": self.gated_layers,
            "gating_hidden_dim": self.gating_network.gating_hidden_dim if hasattr(self.gating_network, 'gating_hidden_dim') else 256,
            "gating_dropout": self.gating_network.dropout_rate if hasattr(self.gating_network, 'dropout_rate') else 0.1,
            "gating_temperature": self.gating_network.temperature if hasattr(self.gating_network, 'temperature') else 1.0,
        }
        torch.save(config, os.path.join(save_directory, "gated_lora_config.pt"))

        logger.info(f"Saved GatedLoRA to {save_directory}")

    def load_adapter_state(self, save_directory: str):
        """Load expert pools + gating network weights IN PLACE (for resume).

        Unlike ``from_pretrained`` this does not rebuild the model (no base
        model reload): it restores only the trainable state into the already
        constructed instance. This is what the trainer needs for cross-job
        SLURM resume — the previous implementation silently skipped model
        weights on resume, so chained jobs restarted from fresh adapters
        while reusing the old optimizer state.
        """
        import os

        expert_path = os.path.join(save_directory, "expert_pools.pt")
        gating_path = os.path.join(save_directory, "gating_network.pt")
        if not os.path.exists(expert_path) or not os.path.exists(gating_path):
            raise FileNotFoundError(
                f"Missing adapter files in {save_directory} "
                f"(expected expert_pools.pt + gating_network.pt)"
            )

        expert_states = torch.load(expert_path, map_location=self.device, weights_only=True)
        for i, pool in enumerate(self.expert_pools):
            pool.load_state_dict(expert_states[f"layer_{i}"])

        gating_state = torch.load(gating_path, map_location=self.device, weights_only=True)
        self.gating_network.load_state_dict(gating_state)

        logger.info(f"Loaded adapter state (experts + gating) from {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory: str, config_override: Dict[str, Any] = None, **kwargs):
        """
        Load model from saved directory.

        Args:
            save_directory: Path to saved model directory
            config_override: Optional dict to override saved config (useful for ablations
                           where config wasn't fully saved, e.g. use_layer_embedding)
            **kwargs: Additional overrides
        """
        import os
        import inspect
        import json

        # Load config from .pt file
        config_path = os.path.join(save_directory, "gated_lora_config.pt")
        config = torch.load(config_path, weights_only=True)

        # Try to load additional config from JSON if it exists (for ablations)
        # This allows us to get params that weren't saved in the .pt file
        json_config_candidates = [
            os.path.join(save_directory, "config.json"),
            os.path.join(os.path.dirname(save_directory), "config.json"),
        ]
        for json_path in json_config_candidates:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_config = json.load(f)
                # Extract model config if nested
                if "model" in json_config:
                    json_config = json_config["model"]
                # Only add keys that are missing from the .pt config
                for key in ["use_layer_embedding", "gated_layers", "gating_hidden_dim",
                           "gating_dropout", "gating_temperature"]:
                    if key not in config and key in json_config:
                        config[key] = json_config[key]
                        logger.info(f"Loaded missing config key '{key}' from JSON: {json_config[key]}")
                break

        # Apply config_override if provided
        if config_override:
            config.update(config_override)
            logger.info(f"Applied config override: {config_override}")

        # Apply kwargs
        config.update(kwargs)

        # Try to infer gating_hidden_dim from saved weights if not in config
        if "gating_hidden_dim" not in config:
            gating_path = os.path.join(save_directory, "gating_network.pt")
            if os.path.exists(gating_path):
                gating_state = torch.load(gating_path, map_location="cpu", weights_only=True)
                # Look for the first layer's gate weight to infer hidden dim
                # Keys are like "gating.layer_gates.0.gate.0.weight" with shape [gating_hidden_dim, input_dim]
                for key in gating_state:
                    # Match patterns like "layer_gates.0.gate.0.weight" or "gating.layer_gates.0.gate.0.weight"
                    if "layer_gates.0.gate.0.weight" in key or key == "gating.gate.gate.0.weight":
                        inferred_dim = gating_state[key].shape[0]
                        config["gating_hidden_dim"] = inferred_dim
                        logger.info(f"Inferred gating_hidden_dim={inferred_dim} from checkpoint weights (key: {key})")
                        break
                else:
                    # Fallback: try any gate.0.weight key
                    for key in gating_state:
                        if key.endswith(".gate.0.weight"):
                            inferred_dim = gating_state[key].shape[0]
                            config["gating_hidden_dim"] = inferred_dim
                            logger.info(f"Inferred gating_hidden_dim={inferred_dim} from checkpoint weights (fallback key: {key})")
                            break

        # Filter to only valid __init__ parameters
        valid_params = inspect.signature(cls.__init__).parameters.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_params}

        logger.info(f"Creating model with config: use_layer_embedding={filtered_config.get('use_layer_embedding', True)}, "
                   f"gated_layers={filtered_config.get('gated_layers', None)}, "
                   f"gating_hidden_dim={filtered_config.get('gating_hidden_dim', 256)}")

        # Create model
        model = cls(**filtered_config)

        # Load expert pools
        expert_states = torch.load(os.path.join(save_directory, "expert_pools.pt"), weights_only=True)
        for i, pool in enumerate(model.expert_pools):
            pool.load_state_dict(expert_states[f"layer_{i}"])

        # Load gating network (with strict=False to handle missing keys gracefully)
        gating_state = torch.load(os.path.join(save_directory, "gating_network.pt"), weights_only=True)
        try:
            model.gating_network.load_state_dict(gating_state, strict=True)
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                logger.warning(f"State dict mismatch, loading with strict=False: {e}")
                model.gating_network.load_state_dict(gating_state, strict=False)
            else:
                raise

        logger.info(f"Loaded GatedLoRA from {save_directory}")
        return model


def create_gated_lora_model(
    model_name: str = "microsoft/phi-2",
    expert_ranks: List[int] = None,
    expert_alphas: List[int] = None,
    target_modules: List[str] = None,
    lora_dropout: float = 0.1,
    gating_hidden_dim: int = 256,
    gating_dropout: float = 0.1,
    per_layer_gating: bool = True,
    use_top_k: bool = False,
    top_k: int = 2,
    gating_temperature: float = 1.0,
    use_layer_embedding: bool = True,  # NEW
    gated_layers: List[int] = None,  # NEW: partial gating
    use_load_balancing: bool = True,
    load_balancing_weight: float = 0.001,
    use_l1_gate_regularization: bool = True,
    l1_gate_weight: float = 0.01,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> GatedLoRAModelV2:
    """
    Factory function to create a GatedLoRAModelV2.

    This is the main entry point for creating gated LoRA models.

    Args:
        model_name: HuggingFace model name
        expert_ranks: List of LoRA ranks for each expert
        expert_alphas: List of LoRA alphas for each expert
        target_modules: Which modules to apply LoRA to
        lora_dropout: Dropout for LoRA layers
        gating_hidden_dim: Hidden dimension for gating MLP
        gating_dropout: Dropout for gating network
        per_layer_gating: Whether to use per-layer gating
        use_top_k: Whether to use sparse top-k routing
        top_k: Number of experts to use if top-k routing
        gating_temperature: Temperature for softmax
        use_load_balancing: Whether to use load balancing loss
        load_balancing_weight: Weight for load balancing loss
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code

    Returns:
        GatedLoRAModelV2 instance
    """
    # Convert string dtype to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(torch_dtype, str):
        torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    return GatedLoRAModelV2(
        model_name=model_name,
        expert_ranks=expert_ranks,
        expert_alphas=expert_alphas,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        gating_hidden_dim=gating_hidden_dim,
        gating_dropout=gating_dropout,
        per_layer_gating=per_layer_gating,
        use_top_k=use_top_k,
        top_k=top_k,
        gating_temperature=gating_temperature,
        use_layer_embedding=use_layer_embedding,  # NEW
        gated_layers=gated_layers,  # NEW
        use_load_balancing=use_load_balancing,
        load_balancing_weight=load_balancing_weight,
        use_l1_gate_regularization=use_l1_gate_regularization,
        l1_gate_weight=l1_gate_weight,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )


if __name__ == "__main__":
    # Test the model
    print("Testing GatedLoRAModelV2...")

    # Small test
    model = GatedLoRAModelV2(
        model_name="microsoft/phi-2",
        expert_ranks=[8, 16],
        expert_alphas=[16, 32],
        target_modules=["q_proj"],
        per_layer_gating=False,  # Simpler for testing
        use_load_balancing=True,
        load_balancing_weight=0.001,
    )

    # Test forward
    input_ids = torch.randint(0, 1000, (2, 64)).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    model.train()
    outputs = model(input_ids, attention_mask, labels, return_routing_info=True)

    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"LM Loss: {outputs['lm_loss'].item():.4f}")
    print(f"LB Loss: {outputs.get('load_balance_loss', 0):.6f}")

    if "routing_info" in outputs:
        print(f"Routing entropy: {outputs['routing_info']['mean_entropy'].item():.4f}")

    stats = model.get_routing_stats()
    print(f"Routing stats: {stats}")

    print(f"\nTrainable params: {model.get_trainable_params()}")
    print("Test passed!")
