"""Output head that projects only the supervised positions.

With answer-only loss most positions have label -100, yet a plain causal-LM forward runs the
vocabulary projection (152k x 896 for Qwen2.5-0.5B, ~27% of its forward FLOPs) and an fp32
cross-entropy over every position. `SelectiveLMHead` wraps the model's output embedding: when
`index` (flat positions into [batch * seq]) is set, it projects only those rows and the model
returns logits of shape [N, vocab] instead of [batch, seq, vocab]. With `index = None` it is the
original head, so generation, analysis scripts and checkpoints are unaffected (adapters only are
saved; the wrapped weight is the same, possibly tied, Parameter).

Valid for HF causal LMs that apply the head once to the final hidden states (Llama, Qwen2,
Gemma-2 incl. its elementwise logit soft-capping, GPT-NeoX, Phi). Call with labels=None and
logits_to_keep left at its default (0 = all positions).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SelectiveLMHead(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.index: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return self.inner.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.inner, "bias", None)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.index is None:
            return self.inner(hidden)
        if hidden.dim() != 3:
            raise RuntimeError(f"SelectiveLMHead expects [batch, seq, hidden], got {tuple(hidden.shape)}")
        return self.inner(hidden.reshape(-1, hidden.shape[-1]).index_select(0, self.index))


def install_selective_head(hf_model: nn.Module) -> SelectiveLMHead:
    """Replace `hf_model`'s output embedding by a SelectiveLMHead (idempotent)."""
    head = hf_model.get_output_embeddings()
    if isinstance(head, SelectiveLMHead):
        return head
    if head is None:
        raise ValueError(f"{type(hf_model).__name__} has no output embeddings")
    for parent in hf_model.modules():
        for name, child in parent.named_children():
            if child is head:
                wrapper = SelectiveLMHead(head)
                setattr(parent, name, wrapper)
                return wrapper
    raise ValueError("Output embedding module not found among the model's children")
