"""Test the multi-task dataset presets are exposed correctly."""

from __future__ import annotations

from gated_lora.data import (
    get_all_8_tasks,
    get_diverse_6_tasks,
    get_harder_4_tasks,
    get_original_4_tasks,
    get_reasoning_focused,
)


def _check_preset(fn):
    cfg = fn()
    assert "tasks" in cfg and "weights" in cfg
    assert len(cfg["tasks"]) == len(cfg["weights"])
    assert all(isinstance(t, str) for t in cfg["tasks"])
    assert abs(sum(cfg["weights"]) - 1.0) < 1e-3
    return cfg


def test_original_4_preset():
    cfg = _check_preset(get_original_4_tasks)
    assert set(cfg["tasks"]) == {"squad", "imdb", "conll2003", "wikitext"}


def test_harder_4_preset():
    cfg = _check_preset(get_harder_4_tasks)
    assert set(cfg["tasks"]) == {"gsm8k", "xsum", "commonsenseqa", "mnli"}


def test_all_8_preset_is_union():
    cfg = _check_preset(get_all_8_tasks)
    expected = {"squad", "imdb", "conll2003", "wikitext", "gsm8k", "xsum", "commonsenseqa", "mnli"}
    assert set(cfg["tasks"]) == expected


def test_reasoning_focused_preset():
    cfg = _check_preset(get_reasoning_focused)
    # Must contain reasoning-heavy tasks
    assert "gsm8k" in cfg["tasks"]
    assert "commonsenseqa" in cfg["tasks"]


def test_diverse_6_preset():
    _check_preset(get_diverse_6_tasks)


# ---- v2 sampler sort window + collated supervised positions (2026-10-07) ----

def _sampler(**kw):
    from gated_lora.data.multi_task_dataset import WeightedTaskBatchSampler
    sizes = [37, 23, 40]
    lengths = [((i * 7919) % 500) + 5 for i in range(sum(sizes))]
    base = dict(task_sizes=sizes, task_weights=[0.5, 0.2, 0.3], num_samples=96, batch_size=4,
                seed=3, epoch=1, lengths=lengths)
    base.update(kw)
    return WeightedTaskBatchSampler(**base), lengths


def test_sort_window_keeps_each_step_mix_and_sorts():
    plain, lengths = _sampler()
    sorted_, _ = _sampler(sort_window=16)
    a = [i for b in plain._build() for i in b]
    bs = sorted_._build()
    for w in range(0, 96, 16):  # same multiset of draws per optimizer step (4 x 4)
        win = [i for b in bs[w // 4:w // 4 + 4] for i in b]
        assert sorted(win) == sorted(a[w:w + 16])
        lens = [lengths[i] for i in win]
        assert lens == sorted(lens, reverse=True)


def test_sort_window_resume_is_exact():
    full, _ = _sampler(sort_window=16)
    flat = [i for b in full._build() for i in b]
    for skip in (0, 4, 6, 16, 50):
        s, _ = _sampler(sort_window=16, skip_samples=skip)
        assert [i for b in s._build() for i in b] == flat[skip:]


def test_sort_window_validation():
    import pytest
    with pytest.raises(ValueError):
        _sampler(sort_window=10)  # not a multiple of batch_size
    with pytest.raises(ValueError):
        _sampler(sort_window=16, length_bucketing=True)


def test_collator_supervised_positions():
    import torch
    from gated_lora.data.multi_task_dataset import Collator
    c = Collator(pad_token_id=0, answer_only_loss=True)
    batch = [{"input_ids": torch.tensor([5, 6, 7, 8, 9]), "prompt_len": 3, "task": "a",
              "example_idx": 11},
             {"input_ids": torch.tensor([5, 6, 7]), "prompt_len": 1, "task": "b",
              "example_idx": 4}]
    out = c(batch)
    T = out["input_ids"].shape[1]
    assert T == 5
    # row 0 answer tokens at 3,4 -> predicted from positions 2,3; row 1 tokens 1,2 -> 0,1
    assert out["sup_flat"].tolist() == [2, 3, T + 0, T + 1]
    assert out["sup_tgt"].tolist() == [8, 9, 6, 7]
    assert out["sup_rows"].tolist() == [0, 0, 1, 1]
    assert out["ans_flat"].tolist() == out["sup_flat"].tolist()  # answer_only_loss
    assert out["example_idx"].tolist() == [11, 4] and out["prompt_len"].tolist() == [3, 1]


# --- val/train/final split construction (offline: fake HF splits) -------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gated_lora.data.multi_task_dataset import MultiTaskDatasetLoader  # noqa: E402

FAKE_SIZES = {"train": 400, "validation": 60, "test": 50, "validation_matched": 40}


class _FakeSplit:
    def __init__(self, name):
        self.name = name

    def __len__(self):
        return FAKE_SIZES[self.name]

    def __getitem__(self, column):  # group columns: 40 groups of 10 rows
        return [f"g{i // 10}" for i in range(len(self))]


def _loader(data_format, seed=0):
    ld = MultiTaskDatasetLoader.__new__(MultiTaskDatasetLoader)
    ld.__dict__.update(data_format=data_format, split_seed=0, seed=seed, val_fraction=0.05,
                       _partition_cache={}, _hf_cache={},
                       # read by cache_key only
                       tokenizer=type("Tok", (), {"__len__": lambda self: 10})(),
                       tokenizer_name="tok", bos_ids=[], eos_id=0, max_length=8,
                       max_samples_per_task=None, max_val_samples=None, max_final_samples=None)
    ld._load_hf = lambda task, split: _FakeSplit(split)
    return ld


def _rows(ld, task, role):
    ds, idx = ld._candidates(task, role)
    return ds.name, set(int(i) for i in idx)


@pytest.mark.parametrize("task", ["wikitext", "conll2003", "xsum"])
def test_v3_heldout_tasks_use_official_val_and_test(task):
    ld = _loader("v3")
    assert _rows(ld, task, "train") == ("train", set(range(400)))
    assert _rows(ld, task, "val") == ("validation", set(range(60)))
    assert _rows(ld, task, "final") == ("test", set(range(50)))
    assert ld.cache_key(task, "val")["hf_split"] == "validation"
    assert ld.cache_key(task, "val")["val_fraction"] is None


@pytest.mark.parametrize("task", ["wikitext", "conll2003", "xsum", "gsm8k", "squad"])
def test_v2_val_is_a_disjoint_train_slice(task):
    ld = _loader("v2")
    tr_split, tr = _rows(ld, task, "train")
    val_split, val = _rows(ld, task, "val")
    assert tr_split == val_split == "train"
    assert not tr & val and len(tr | val) == 400
    assert _rows(ld, task, "final")[0] != "train"


@pytest.mark.parametrize("task", ["squad", "imdb", "gsm8k", "commonsenseqa", "mnli"])
def test_v3_keeps_v2_splits_for_other_tasks(task):
    for role in ("train", "val", "final"):
        assert _rows(_loader("v3"), task, role) == _rows(_loader("v2"), task, role)


def test_v2_and_v3_never_share_a_cache_entry():
    ld2, ld3 = _loader("v2"), _loader("v3")
    for task in ("gsm8k", "wikitext"):
        for role in ("train", "val", "final"):
            assert ld2.cache_key(task, role) != ld3.cache_key(task, role)


def test_v3_train_order_depends_on_run_seed_only():
    a, b, c = _loader("v3", seed=1), _loader("v3", seed=1), _loader("v3", seed=2)
    ia, ib, ic = (ld._candidates("wikitext", "train")[1] for ld in (a, b, c))
    assert np.array_equal(ia, ib) and not np.array_equal(ia, ic)
    assert np.array_equal(a._candidates("wikitext", "val")[1], c._candidates("wikitext", "val")[1])
