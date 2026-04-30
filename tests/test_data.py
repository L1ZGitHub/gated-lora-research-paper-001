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
