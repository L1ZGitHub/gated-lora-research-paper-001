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
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig

from .lora_experts import LoRAExpertPool, FUSED_EXPERTS
from .gating_network import GatingNetwork, LayerGatingNetwork

logger = logging.getLogger(__name__)


_DTYPE_ALIASES = {
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
}


def resolve_precision(
    precision: Optional[Union[str, torch.dtype]],
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    """
    Resolve the base-model dtype.

    precision: "auto" | "fp16" | "bf16" | "fp32" (aliases float16/bfloat16/float32 accepted).
    "auto" = bf16 if the current CUDA device has compute capability >= (8, 0), fp16 on older
    GPUs (e.g. sm_75: bf16 has no fast matmul / memory-efficient SDPA there), fp32 without
    CUDA. If precision is None, the legacy ``torch_dtype`` is used (None -> "auto").
    """
    if precision is None:
        if torch_dtype is None:
            precision = "auto"
        elif isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        else:
            precision = torch_dtype
    if isinstance(precision, torch.dtype):
        return precision
    key = str(precision).strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
        return torch.float32
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unknown precision {precision!r}; expected auto | fp16 | bf16 | fp32")
    return _DTYPE_ALIASES[key]


def _resolve_gate_output_scale(value: Union[str, float, None], num_experts: int) -> float:
    """"num_experts" -> float(num_experts); numbers (or numeric strings) -> float; None -> 1.0."""
    if value is None:
        return 1.0
    if isinstance(value, str) and value.strip().lower() == "num_experts":
        return float(num_experts)
    scale = float(value)
    if not scale > 0:
        raise ValueError(f"gate_output_scale must be > 0 or 'num_experts', got {value!r}")
    return scale


def _resolve_gated_layers(
    gated_layers: Optional[List[int]],
    gated_layers_frac: Optional[List[float]],
    num_layers: int,
) -> Optional[List[int]]:
    """
    Resolve partial gating to a sorted list of layer indices (None = all layers).

    gated_layers_frac = [start, end) as fractions of depth; layer l is gated iff
    round(start * L) <= l < round(end * L) (e.g. [0.75, 1.0] with L=32 -> 24..31).
    Same rule as training/config.py:resolve_gated_layers (python round = half-to-even);
    adjacent fraction ranges partition the layers without gap or overlap.
    """
    if gated_layers is not None and gated_layers_frac is not None:
        raise ValueError("Set only one of gated_layers and gated_layers_frac")

    if gated_layers_frac is not None:
        if len(gated_layers_frac) != 2:
            raise ValueError(f"gated_layers_frac must be [start, end), got {gated_layers_frac}")
        start, end = float(gated_layers_frac[0]), float(gated_layers_frac[1])
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(f"gated_layers_frac must satisfy 0 <= start < end <= 1, got {gated_layers_frac}")
        lo = int(round(start * num_layers))
        hi = int(round(end * num_layers))
        gated_layers = list(range(lo, hi))
        if not gated_layers:
            raise ValueError(
                f"gated_layers_frac={gated_layers_frac} selects no layer out of {num_layers}"
            )

    if gated_layers is None:
        return None

    resolved = sorted({int(i) for i in gated_layers})
    bad = [i for i in resolved if not (0 <= i < num_layers)]
    if bad:
        raise ValueError(f"gated_layers {bad} out of range for a model with {num_layers} layers")
    if not resolved:
        logger.warning("gated_layers is empty: every layer uses uniform routing (no gate is trained)")
    return resolved


def _autocast_enabled(device_type: str) -> bool:
    """torch.is_autocast_enabled for a device type, across torch versions."""
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # torch < 2.4: no device_type argument
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return torch.is_autocast_enabled()


def _fp32_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detached copy of a state dict with floating tensors in fp32 (save format)."""
    return {k: (v.detach().float() if v.is_floating_point() else v.detach()) for k, v in state.items()}


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
        use_layer_embedding: bool = True,  # Only used with per_layer_gating=False
        gated_layers: List[int] = None,  # Only apply gating to these layers (None = all)
        gated_layers_frac: Optional[List[float]] = None,  # [start, end) fractions of depth
        gate_output_scale: Union[str, float] = "num_experts",
        # Load balancing
        use_load_balancing: bool = True,
        load_balancing_weight: float = 0.001,
        # L1 regularization on gates (per Gated LoRA spec; implemented as entropy)
        use_l1_gate_regularization: bool = True,
        l1_gate_weight: float = 0.01,
        # Model loading
        precision: Optional[str] = "auto",
        torch_dtype: Optional[Union[torch.dtype, str]] = None,  # legacy, used only if precision is None
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

        if gated_layers is not None and gated_layers_frac is not None:
            raise ValueError(
                f"Set only one of gated_layers ({gated_layers}) and gated_layers_frac "
                f"({gated_layers_frac})."
            )

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
        self.gated_layers_frac = gated_layers_frac

        # gate_output_scale: the gate weights sum to 1 over experts, so with scale 1 each
        # expert's delta is effectively multiplied by ~1/num_experts compared with a plain
        # LoRA at the same alpha/r. With "num_experts", uniform routing (g_e = 1/E) gives
        # exactly sum_e s_e B_e A_e x, i.e. a plain LoRA with the summed ranks (each block at
        # its own alpha/r). Legacy checkpoints were trained with 1.0 (from_pretrained sets it).
        self.gate_output_scale = gate_output_scale
        self.gate_output_scale_value = _resolve_gate_output_scale(gate_output_scale, self.num_experts)

        # Trainer-controlled knobs (see docs/v2_contract.md)
        self.reg_scale: float = 1.0  # multiplies the entropy ("L1") regulariser, 0->1 ramp
        self.collect_routing_stats: bool = False  # routing stats only when True or return_routing_info

        # Precision: base model dtype. Trainable params (experts, gates) are always fp32.
        self.precision = precision
        base_dtype = resolve_precision(precision, torch_dtype)
        if precision is not None and torch_dtype is not None:
            logger.info(f"  torch_dtype={torch_dtype} ignored (precision={precision!r} is set)")
        self.adapter_dtype = torch.float32

        logger.info(f"Initializing GatedLoRAModelV2 with {self.num_experts} experts")
        logger.info(f"  Ranks: {expert_ranks}")
        logger.info(f"  Alphas: {expert_alphas}")
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Precision: {precision!r} -> base model {base_dtype}, adapters {self.adapter_dtype}")
        logger.info(f"  gate_output_scale: {gate_output_scale!r} -> {self.gate_output_scale_value}")

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
            torch_dtype=base_dtype,
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

        # Resolve partial gating (indices, bounds-checked). None means all layers.
        self.gated_layers = _resolve_gated_layers(gated_layers, gated_layers_frac, self.num_layers)
        self._gated_layer_set = None if self.gated_layers is None else frozenset(self.gated_layers)

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

        # Create gating network (it ignores use_layer_embedding with per-layer gates and
        # logs a warning; the effective value is read back below)
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
            use_layer_embedding=use_layer_embedding,
        )
        self.use_layer_embedding = self.gating_network.use_layer_embedding

        # Log gated layers config
        if self.gated_layers is not None:
            logger.info(f"  Partial gating: only layers {self.gated_layers}"
                        + (f" (from gated_layers_frac={gated_layers_frac})" if gated_layers_frac is not None else ""))
        else:
            logger.info(f"  Full gating: all {self.num_layers} layers")

        # Device and dtype
        self.device = next(self.model.parameters()).device
        self.dtype = base_dtype  # base model dtype (adapters: self.adapter_dtype)

        # Experts and gating stay fp32 whatever the base precision
        for pool in self.expert_pools:
            pool.to(device=self.device, dtype=self.adapter_dtype)
        self.gating_network.to(device=self.device, dtype=self.adapter_dtype)

        # Per-forward state (reset at the start of every forward)
        self._routing_info_per_layer: Dict[int, Dict] = {}
        self._accumulated_load_balance_loss = 0.0
        self._accumulated_l1_gate_loss = 0.0
        self._num_lb_layers = 0
        self._num_entropy_layers = 0
        self._stats_this_forward = False
        self._token_mask: Optional[torch.Tensor] = None
        self._token_mask_by_device: Dict[torch.device, torch.Tensor] = {}
        # Per-layer cache: layer_idx -> (gate_weights, expanded_gate). Computed by the first
        # hooked module of the layer and reused by the others. Cleared after the base forward.
        self._gating_cache: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

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

        # Per-layer gating is cached for the duration of ONE base-model call: reset it at the
        # start of every call, so that direct calls of the base model (HF generate with a KV
        # cache: one call per new token) never reuse a stale gate.
        self._hooks.append(self.model.register_forward_pre_hook(self._reset_layer_caches))

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
        """
        Create a forward hook for a specific module in a layer.

        dtype handling: the base Linear runs in the base precision (fp16/bf16/fp32) while the
        experts and gates are fp32. Without autocast, the Linear input is cast to the adapter
        dtype for the gate + LoRA math. Under autocast it is left as is (F.linear autocasts
        the fp32 weights itself; an explicit fp32 copy would be wasted). The delta is always
        cast back to the Linear output dtype before the residual add.
        """
        def hook(module, inputs, outputs):
            # inputs[0] is the input to the Linear: [batch, seq, in_dim]
            x = self._to_adapter_dtype(inputs[0])

            # All modules in a layer share the same gating, computed once per layer
            # (by the first hooked module, which must see hidden_size inputs).
            cached = self._gating_cache.get(layer_idx)
            if cached is None:
                cached = self._compute_layer_gating(layer_idx, module_name, x)
            gate_weights, expanded_gate = cached

            # Apply LoRA for THIS specific module. With top-k, gate_weights are the
            # tempered sparse weights (zeros for unselected experts).
            lora_delta = self.expert_pools[layer_idx].get_weighted_output(
                x,
                module_name=module_name,
                gate_weights=gate_weights,
                output_scale=self.gate_output_scale_value,
                expanded_gate=expanded_gate,
            )

            # Add LoRA delta to the Linear output (in the Linear output dtype)
            return outputs + lora_delta.to(outputs.dtype)

        return hook

    def _reset_layer_caches(self, module=None, args=None) -> None:
        self._gating_cache = {}
        self._token_mask_by_device = {}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """HF generate on the base model with the experts applied (eval only: no regularisers,
        no routing stats; gates computed per call on the new tokens)."""
        self._check_no_gradient_checkpointing()
        prev = (self._token_mask, self._stats_this_forward)
        self._token_mask, self._stats_this_forward = None, False
        self._accumulated_load_balance_loss = 0.0
        self._accumulated_l1_gate_loss = 0.0
        self._num_lb_layers = self._num_entropy_layers = 0
        try:
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            self._reset_layer_caches()
            self._token_mask, self._stats_this_forward = prev

    def _to_adapter_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Cast an activation to the adapter dtype unless autocast is active (see hook docstring)."""
        if x.dtype == self.adapter_dtype or _autocast_enabled(x.device.type):
            return x
        return x.to(self.adapter_dtype)

    def _token_mask_on(self, device: torch.device) -> Optional[torch.Tensor]:
        """Real-token mask [batch, seq] (float) on ``device``; None if no mask was given."""
        if self._token_mask is None:
            return None
        mask = self._token_mask_by_device.get(device)
        if mask is None:
            mask = self._token_mask.to(device, non_blocking=True)
            self._token_mask_by_device[device] = mask
        return mask

    def _compute_layer_gating(
        self,
        layer_idx: int,
        module_name: str,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute (once per layer) the gate weights, the regularisers and, if requested, the
        routing stats. Caches and returns (gate_weights, expanded_gate).
        """
        # Only compute gating if input dimension matches hidden_size
        # For GQA models, o_proj input has different dimension (num_heads * head_dim)
        input_dim = x.shape[-1]
        if input_dim != self.hidden_size:
            # This should not happen in normal execution order (q/k/v come before o)
            raise RuntimeError(
                f"Gating cache miss for layer {layer_idx}, module {module_name}. "
                f"Input dim {input_dim} != hidden_size {self.hidden_size}. "
                f"This may indicate hooks are firing in unexpected order."
            )

        token_mask = self._token_mask_on(x.device)
        use_gating_for_layer = self._gated_layer_set is None or layer_idx in self._gated_layer_set

        if use_gating_for_layer:
            gate_weights, gate_logits, routing_info = self.gating_network.compute_gate_weights(
                x, layer_idx, token_mask=token_mask, compute_stats=self._stats_this_forward
            )

            if self.training and self.use_load_balancing:
                lb_loss = self.gating_network.compute_load_balancing_loss(
                    gate_weights, gate_logits, token_mask=token_mask
                )
                self._accumulated_load_balance_loss = self._accumulated_load_balance_loss + lb_loss
                self._num_lb_layers += 1

            # Gate sparsity regulariser ("L1" for historical config names): mean entropy of
            # the gate distribution over real tokens. NOTE (2026-07 fix): the historical
            # `gate_weights.abs().mean()` was a no-op post-softmax (constant 1/num_experts).
            if self.training and self.use_l1_gate_regularization:
                gate_entropy = self.gating_network.compute_gate_entropy(gate_weights, token_mask)
                self._accumulated_l1_gate_loss = self._accumulated_l1_gate_loss + gate_entropy
                self._num_entropy_layers += 1
        else:
            # Uniform routing for non-gated layers (partial gating ablation)
            batch_size, seq_len = x.shape[:2]
            gate_weights = torch.full(
                (batch_size, seq_len, self.num_experts),
                1.0 / self.num_experts,
                device=x.device,
                dtype=self.adapter_dtype,
            )
            routing_info = {"gate_weights": gate_weights, "uniform": True}

        if self._stats_this_forward:
            self._routing_info_per_layer[layer_idx] = routing_info

        # Per-rank gate expansion, shared by every target module of the layer (fused path)
        expanded_gate = None
        if FUSED_EXPERTS:
            expanded_gate = self.expert_pools[layer_idx].expand_gate_weights(
                gate_weights, self.gate_output_scale_value
            )

        cached = (gate_weights, expanded_gate)
        self._gating_cache[layer_idx] = cached
        return cached

    def _check_no_gradient_checkpointing(self):
        """The per-layer gating cache is incompatible with activation recomputation."""
        if self.training and getattr(self.model, "is_gradient_checkpointing", False):
            raise RuntimeError(
                "Gradient checkpointing is enabled on the base model, which is incompatible "
                "with GatedLoRAModelV2: the per-layer gating cache (and the regulariser "
                "accumulators) are filled by forward hooks, and recomputation in backward would "
                "re-run the hooks against a stale/cleared cache. Disable gradient checkpointing "
                "(model.model.gradient_checkpointing_disable())."
            )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Not supported (see ``_check_no_gradient_checkpointing``)."""
        raise RuntimeError(
            "GatedLoRAModelV2 does not support gradient checkpointing: the per-layer gating "
            "cache filled by forward hooks would be stale on recompute."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
        base_attention_mask: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gated LoRA.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]; also used so that the load-balancing loss,
                the entropy regulariser and the routing stats only count real tokens
            labels: [batch, seq_len] for loss computation
            return_routing_info: Whether to return detailed routing info (forces the
                routing stats for this forward, like ``self.collect_routing_stats``)
            base_attention_mask: pass ``attention_mask`` to the base model. False is exact ONLY
                for RIGHT-padded batches (causal attention never reaches the padding, and
                position ids do not depend on the mask): the base model then uses plain
                causal attention (no 4D mask); the hooks still use the mask.
            **kwargs: Additional arguments for base model (use_cache defaults to False)

        Returns:
            Dict with loss, lm_loss, logits, load_balancing_loss, l1_gate_loss (training
            with labels) and routing_info (if requested). The two regularisers are MEANS over
            the gated layers; l1_gate_loss is the raw mean entropy (before l1_gate_weight and
            reg_scale).
        """
        self._check_no_gradient_checkpointing()

        # Reset accumulated losses and caches
        self._accumulated_load_balance_loss = 0.0
        self._accumulated_l1_gate_loss = 0.0
        self._num_lb_layers = 0
        self._num_entropy_layers = 0
        self._routing_info_per_layer = {}
        self._gating_cache = {}
        self._token_mask_by_device = {}
        self._stats_this_forward = bool(self.collect_routing_stats or return_routing_info)

        # Real-token mask for the hooks (shape check is on python ints: no host sync)
        self._token_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2 and tuple(attention_mask.shape) == tuple(input_ids.shape):
                self._token_mask = attention_mask.to(dtype=self.adapter_dtype)
            elif not getattr(self, "_warned_mask_shape", False):
                logger.warning(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                    f"input_ids {tuple(input_ids.shape)}: routing regularisers/stats use all tokens."
                )
                self._warned_mask_shape = True

        kwargs.setdefault("use_cache", False)

        # Forward through base model (hooks will apply LoRA)
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask if base_attention_mask else None,
                labels=labels,
                **kwargs,
            )
        finally:
            # Hooks are done: drop the per-layer cache (autograd keeps what backward needs)
            self._gating_cache = {}
            self._token_mask_by_device = {}

        # Build result
        result = {
            "logits": outputs.logits,
        }

        if labels is not None:
            lm_loss = outputs.loss
            aux = self.aux_losses(lm_loss.device)
            total_loss = lm_loss + aux.pop("total")
            result.update(aux)
            result["loss"] = total_loss
            result["lm_loss"] = lm_loss

        # Add routing info
        if return_routing_info and self._routing_info_per_layer:
            result["routing_info"] = self._aggregate_routing_info()

        return result

    def aux_losses(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Routing regularisers accumulated by the hooks during the LAST forward (training mode only).

        Returns ``total`` (weighted sum to add to the LM loss) plus the raw
        ``load_balancing_loss`` and ``l1_gate_loss`` (mean entropy) when enabled. Lets a caller
        that computes the LM loss itself (selective output head) add the same regularisers as
        ``forward(labels=...)`` does.
        """
        zero = torch.zeros((), device=device, dtype=torch.float32)
        out: Dict[str, torch.Tensor] = {"total": zero}
        if self.use_load_balancing and self.training:
            lb_loss = (self._accumulated_load_balance_loss / self._num_lb_layers
                       if self._num_lb_layers > 0 else zero)
            out["total"] = out["total"] + self.load_balancing_weight * lb_loss
            out["load_balancing_loss"] = lb_loss
        if self.use_l1_gate_regularization and self.training:
            entropy_loss = (self._accumulated_l1_gate_loss / self._num_entropy_layers
                            if self._num_entropy_layers > 0 else zero)
            out["total"] = out["total"] + (self.l1_gate_weight * self.reg_scale) * entropy_loss
            out["l1_gate_loss"] = entropy_loss
        return out

    def _aggregate_routing_info(self) -> Dict[str, Any]:
        """
        Aggregate routing info across all layers.

        Per-layer stats are already means over real tokens. ``per_layer_info[l]["gate_weights"]``
        is the full [batch, seq, num_experts] tensor (padding included): use ``token_mask``
        ([batch, seq] float, or None) to exclude padding downstream.
        """
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
                "token_mask": self._token_mask,
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
            "token_mask": self._token_mask,
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

        # Save expert pools (fp32; parameter names unchanged vs legacy checkpoints)
        torch.save(
            {f"layer_{i}": _fp32_state_dict(pool.state_dict()) for i, pool in enumerate(self.expert_pools)},
            os.path.join(save_directory, "expert_pools.pt")
        )

        # Save gating network (fp32)
        torch.save(
            _fp32_state_dict(self.gating_network.state_dict()),
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
            # Layer embedding (effective value) and partial gating (resolved indices)
            "use_layer_embedding": self.use_layer_embedding,
            "gated_layers": self.gated_layers,
            # Informational only (not an __init__ argument, so not re-applied on load:
            # gated_layers above is authoritative)
            "gated_layers_frac_source": self.gated_layers_frac,
            # v2 fields
            "precision": self.precision if self.precision is not None else {
                torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"
            }.get(self.dtype, "auto"),
            "resolved_base_dtype": str(self.dtype),
            "gate_output_scale": self.gate_output_scale,
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

        # load_state_dict copies into the existing fp32 params (casts legacy bf16 weights)
        expert_states = torch.load(expert_path, map_location=self.device, weights_only=True)
        for i, pool in enumerate(self.expert_pools):
            pool.load_state_dict(expert_states[f"layer_{i}"])

        gating_state = torch.load(gating_path, map_location=self.device, weights_only=True)
        gating_state = self._adapt_legacy_gating_state(gating_state)
        self.gating_network.load_state_dict(gating_state)

        logger.info(f"Loaded adapter state (experts + gating) from {save_directory}")

    def _adapt_legacy_gating_state(self, gating_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fold legacy per-layer layer-embedding weights into the first gate bias.

        Legacy per-layer gates computed Linear1(x + e_l) with their own layer index l only,
        i.e. W1 x + (b1 + W1 e_l): the embedding is a pure bias shift. v2 does not allocate
        it with per-layer gating, so the key is dropped and b1 <- b1 + W1 e_l, which keeps
        the gate logits identical up to fp rounding.
        """
        if not isinstance(self.gating_network.gating, LayerGatingNetwork):
            return gating_state
        emb_keys = [k for k in gating_state if k.endswith(".layer_embedding.weight")]
        if not emb_keys:
            return gating_state

        state = dict(gating_state)
        folded = 0
        for key in emb_keys:
            prefix = key[: -len("layer_embedding.weight")]  # "gating.layer_gates.{i}."
            layer_idx = int(prefix.rstrip(".").split(".")[-1])
            emb = state.pop(key)[layer_idx].float()
            w_key, b_key = prefix + "gate.0.weight", prefix + "gate.0.bias"
            if w_key in state and b_key in state:
                w = state[w_key].float()
                state[b_key] = state[b_key].float() + w @ emb.to(w.device)
                folded += 1
        logger.warning(
            f"Legacy checkpoint has layer-embedding weights for {len(emb_keys)} per-layer gates; "
            f"v2 does not use a layer embedding with per-layer gating (it is only a bias shift). "
            f"Dropped the keys and folded W1 @ e_l into the first gate bias for {folded} gates "
            f"(gate logits unchanged up to fp rounding)."
        )
        return state

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

        # Legacy checkpoints (no v2 fields) were trained without output scaling
        if "gate_output_scale" not in config:
            config["gate_output_scale"] = 1.0
            logger.info("Legacy checkpoint: gate_output_scale not saved -> 1.0 (legacy behaviour)")

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

        # Load expert pools (cast into the fp32 params by load_state_dict)
        expert_states = torch.load(os.path.join(save_directory, "expert_pools.pt"),
                                   map_location=model.device, weights_only=True)
        for i, pool in enumerate(model.expert_pools):
            pool.load_state_dict(expert_states[f"layer_{i}"])

        # Load gating network (with strict=False to handle missing keys gracefully)
        gating_state = torch.load(os.path.join(save_directory, "gating_network.pt"),
                                  map_location=model.device, weights_only=True)
        gating_state = model._adapt_legacy_gating_state(gating_state)
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
    use_layer_embedding: bool = True,  # only with per_layer_gating=False
    gated_layers: List[int] = None,  # partial gating (indices)
    gated_layers_frac: Optional[List[float]] = None,  # partial gating ([start, end) depth fractions)
    gate_output_scale: Union[str, float] = "num_experts",
    use_load_balancing: bool = True,
    load_balancing_weight: float = 0.001,
    use_l1_gate_regularization: bool = True,
    l1_gate_weight: float = 0.01,
    precision: Optional[str] = "auto",
    torch_dtype: Optional[str] = None,  # legacy; used only when precision is None
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
        gated_layers / gated_layers_frac: partial gating (at most one of them)
        gate_output_scale: "num_experts" or a float multiplying the mixed expert output
        precision: base model dtype, "auto" | "fp16" | "bf16" | "fp32"
        torch_dtype: legacy base dtype, used only when precision is None
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code

    Returns:
        GatedLoRAModelV2 instance
    """
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
        use_layer_embedding=use_layer_embedding,
        gated_layers=gated_layers,
        gated_layers_frac=gated_layers_frac,
        gate_output_scale=gate_output_scale,
        use_load_balancing=use_load_balancing,
        load_balancing_weight=load_balancing_weight,
        use_l1_gate_regularization=use_l1_gate_regularization,
        l1_gate_weight=l1_gate_weight,
        precision=precision,
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
