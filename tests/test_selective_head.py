"""SelectiveLMHead: logits at selected positions == full-sequence logits; loss == HF loss.

Tiny randomly initialised models built from configs (no download), CPU, fp32.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from gated_lora.models.selective_head import SelectiveLMHead, install_selective_head  # noqa: E402
from gated_lora.training.gated_trainer import GatedLoRATrainer  # noqa: E402

_COMMON = dict(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
               num_attention_heads=4)


def _tiny(kind: str):
    if kind == "qwen2":
        cfg = transformers.Qwen2Config(num_key_value_heads=2, tie_word_embeddings=True, **_COMMON)
    elif kind == "gpt_neox":
        cfg = transformers.GPTNeoXConfig(**_COMMON)
    elif kind == "gemma2":
        cfg = transformers.Gemma2Config(num_key_value_heads=2, head_dim=8,
                                        final_logit_softcapping=30.0, **_COMMON)
    else:
        raise ValueError(kind)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg).eval()


def _batch(T=12, B=3):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 97, (B, T), generator=g)
    mask = torch.ones(B, T, dtype=torch.long)
    if B > 1:
        mask[1, 9:] = 0  # right padding
    labels = torch.full((B, T), -100)
    for r, (a, b) in enumerate([(5, T), (3, 9), (10, T)][:B]):
        labels[r, a:b] = ids[r, a:b]
    answer = labels != -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "answer_mask": answer}


@pytest.mark.parametrize("kind", ["qwen2", "gpt_neox", "gemma2"])
def test_selected_logits_and_loss_match_full_forward(kind):
    model = _tiny(kind)
    b = _batch()
    with torch.no_grad():
        full = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     labels=b["labels"], use_cache=False)
    head = install_selective_head(model)
    assert isinstance(model.get_output_embeddings(), SelectiveLMHead)
    assert install_selective_head(model) is head  # idempotent

    flat, tgt, rows = GatedLoRATrainer._supervised_positions(b, answer_only=False)
    T = b["labels"].shape[1]
    head.index = flat
    try:
        with torch.no_grad():
            sel = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        use_cache=False).logits
    finally:
        head.index = None
    assert sel.shape == (flat.numel(), 97)
    ref = full.logits.reshape(-1, 97)[flat]
    torch.testing.assert_close(sel, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(F.cross_entropy(sel, tgt), full.loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rows, flat // T)

    # index=None restores the original head
    with torch.no_grad():
        again = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                      use_cache=False).logits
    torch.testing.assert_close(again, full.logits)


def test_gradients_flow_through_selected_positions():
    model = _tiny("qwen2").train()
    for p in model.parameters():
        p.requires_grad_(False)
    probe = model.model.layers[-1].mlp.down_proj.weight
    probe.requires_grad_(True)
    b = _batch()
    head = install_selective_head(model)
    flat, tgt, _ = GatedLoRATrainer._supervised_positions(b, answer_only=False)
    head.index = flat
    try:
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       use_cache=False).logits
    finally:
        head.index = None
    F.cross_entropy(logits, tgt).backward()
    assert probe.grad is not None and probe.grad.abs().sum() > 0


def test_supervised_positions_shift_and_answer_mask():
    b = _batch(T=6, B=2)
    b["labels"][:] = -100
    b["labels"][0, 4:] = torch.tensor([7, 8])   # tokens 4,5 supervised -> predicted at 3,4
    b["labels"][1, 2] = 5                       # token 2 -> predicted at position 1
    b["answer_mask"] = b["labels"] != -100
    b["answer_mask"][0, 5] = False               # answer_only drops token 5
    flat, tgt, rows = GatedLoRATrainer._supervised_positions(b, answer_only=False)
    assert flat.tolist() == [3, 4, 6 + 1] and tgt.tolist() == [7, 8, 5] and rows.tolist() == [0, 0, 1]
    flat, tgt, _ = GatedLoRATrainer._supervised_positions(b, answer_only=True)
    assert flat.tolist() == [3, 7] and tgt.tolist() == [7, 5]
