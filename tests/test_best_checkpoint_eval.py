"""Best-checkpoint test eval: control flow of GatedLoRATrainer._evaluate_best_checkpoint
(stubbed trainer, no model): weights swapped to best_model/ and always restored."""

from __future__ import annotations

import json

import pytest

from gated_lora.training.gated_trainer import GatedLoRATrainer, TrainingState


class _Stub(GatedLoRATrainer):
    def __init__(self, out, best_step, global_step, fits=True, fail=False):
        self.output_dir = out
        self.state = TrainingState(global_step=global_step, best_eval_step=best_step)
        self.generation_tasks = ["gsm8k"]
        self.calls = []
        self._fits, self._fail = fits, fail

    def load_weights(self, d):
        self.calls.append(("load", d.name))

    def evaluate(self, which, samples_per_task, dump_path=None):
        self.calls.append(("eval", which, dump_path.name))
        if self._fail:
            raise RuntimeError("boom")
        return {"mean_task_answer_loss": 0.5}

    def generation_eval(self, which):
        self.calls.append(("gen", which))
        return {"gsm8k": {"mean": 0.3, "per_example": [1, 0], "metric": "acc"}}

    def _eval_loader(self, which, n):
        return None

    def _eval_seconds_estimate(self, loader):
        return 60.0

    def _generation_seconds_estimate(self):
        return 300.0

    def time_allows(self, extra_seconds=0.0):
        return self._fits

    def _log_eval(self, metrics, tag):
        pass


FINAL = {"mean_task_answer_loss": 0.6}
GEN = {"gsm8k": {"mean": 0.2, "per_example": [0, 1], "metric": "acc"}}


def test_best_differs_evaluates_best_then_restores_final(tmp_path):
    (tmp_path / "best_model").mkdir()
    t = _Stub(tmp_path, best_step=3000, global_step=5000)
    r = t._evaluate_best_checkpoint(FINAL, GEN)
    assert t.calls == [("load", "best_model"), ("eval", "final", "final_examples_best.npz"),
                       ("gen", "final"), ("load", "final_model")]
    assert r["step"] == 3000 and not r["same_as_final"]
    assert r["final"]["mean_task_answer_loss"] == 0.5
    assert r["generation"] == {"gsm8k": {"mean": 0.3, "metric": "acc"}}  # per-example dropped
    saved = json.load(open(tmp_path / "generation_results_best.json"))
    assert saved["gsm8k"]["per_example"] == [1, 0]


def test_best_is_last_step_reuses_final_results(tmp_path):
    t = _Stub(tmp_path, best_step=5000, global_step=5000)
    r = t._evaluate_best_checkpoint(FINAL, GEN)
    assert t.calls == []
    assert r["same_as_final"] and r["final"] is FINAL
    assert r["generation"] == {"gsm8k": {"mean": 0.2, "metric": "acc"}}


@pytest.mark.parametrize("make_dir,fits", [(False, True), (True, False)])
def test_skipped_without_checkpoint_or_time(tmp_path, make_dir, fits):
    if make_dir:
        (tmp_path / "best_model").mkdir()
    t = _Stub(tmp_path, best_step=3000, global_step=5000, fits=fits)
    assert t._evaluate_best_checkpoint(FINAL, GEN) is None
    assert t.calls == []


def test_eval_failure_is_non_fatal_and_restores_final(tmp_path):
    (tmp_path / "best_model").mkdir()
    t = _Stub(tmp_path, best_step=3000, global_step=5000, fail=True)
    assert t._evaluate_best_checkpoint(FINAL, GEN) is None  # non-fatal
    assert t.calls[-1] == ("load", "final_model")


def test_force_hook_reloads_even_when_best_is_last(tmp_path, monkeypatch):
    monkeypatch.setenv("GLR_BEST_EVAL_ALWAYS", "1")
    (tmp_path / "best_model").mkdir()
    t = _Stub(tmp_path, best_step=5000, global_step=5000)
    r = t._evaluate_best_checkpoint(FINAL, GEN)
    assert t.calls[0] == ("load", "best_model") and t.calls[-1] == ("load", "final_model")
    assert not r["same_as_final"]
