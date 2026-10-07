"""Scoring of greedy generations (GSM8K final number, XSum ROUGE-L)."""

from __future__ import annotations

import pytest

from gated_lora.training.generation_eval import gsm8k_correct, gsm8k_number, rouge_l_f1, score


def test_gsm8k_number_prefers_hash_marker():
    assert gsm8k_number("She has 3 apples. 3 + 4 = 7\n#### 7") == 7.0
    assert gsm8k_number("so 1,250 dollars #### 1,250 then 3") == 1250.0
    assert gsm8k_number("no marker, the answer is 42.") == 42.0
    assert gsm8k_number("nothing") is None


def test_gsm8k_correct():
    assert gsm8k_correct("... #### 18", "Janet ... = 18\n#### 18")
    assert not gsm8k_correct("... #### 17", "#### 18")
    assert not gsm8k_correct("", "#### 18")


def test_rouge_l():
    assert rouge_l_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l_f1("", "x") == 0.0
    # LCS "the cat" (2): P = 2/3, R = 2/4 -> F = 4/7
    assert rouge_l_f1("the big cat", "the cat is here") == pytest.approx(4 / 7)


def test_score_shapes():
    r = score("gsm8k", ["#### 1", "#### 2"], ["#### 1", "#### 3"])
    assert r["mean"] == 0.5 and r["per_example"] == [1.0, 0.0]
    with pytest.raises(ValueError):
        score("imdb", ["a"], ["b"])
