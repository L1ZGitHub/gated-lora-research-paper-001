"""End-of-run HF push: one atomic commit incl. TRAINING_DONE; 429 waits instead of failing."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from gated_lora.training.gated_trainer import TRAINING_DONE, _HubPusher  # noqa: E402


class _FakeApi:
    def __init__(self, fail_429: int = 0):
        self.commits = []
        self.fail_429 = fail_429
        self.files = set()

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.fail_429 > 0:
            self.fail_429 -= 1
            raise RuntimeError("429 Too Many Requests: you have reached your 'api' rate limit.")
        self.commits.append([op.path_in_repo for op in operations])
        self.files.update(op.path_in_repo for op in operations)

    def file_exists(self, repo_id, path, repo_type):
        return path in self.files


def _outputs(tmp: Path) -> Path:
    out = tmp / "run"
    for d in ("final_model", "latest", "best_model"):
        (out / d).mkdir(parents=True)
        (out / d / "w.pt").write_text("x")
    (out / "latest" / "sub").mkdir()
    (out / "latest" / "sub" / "s.json").write_text("{}")
    (out / "eval_results.json").write_text("{}")
    (out / TRAINING_DONE).write_text("done")
    return out


def test_final_push_is_one_commit_with_training_done(tmp_path, monkeypatch):
    out = _outputs(tmp_path)
    p = _HubPusher(out, "repo", "runX", enabled=True)
    api = _FakeApi()
    p._api = api
    ok = p.commit_final(["best_model", "final_model", "latest", "visualizations"],
                        [out / "eval_results.json", out / "missing.json", out / TRAINING_DONE],
                        budget_s=10)
    p.shutdown()
    assert ok and len(api.commits) == 1
    files = set(api.commits[0])
    assert {"runX/final_model/w.pt", "runX/latest/sub/s.json", "runX/eval_results.json",
            f"runX/{TRAINING_DONE}"} <= files
    assert not (out / ".push").exists() or not any((out / ".push").iterdir())


def test_rate_limit_is_waited_out_within_budget(tmp_path, monkeypatch):
    out = _outputs(tmp_path)
    p = _HubPusher(out, "repo", "runX", enabled=True)
    p.RATE_LIMIT_DELAY = 0.01
    api = _FakeApi(fail_429=6)  # more failures than RETRIES: only the budget may stop it
    p._api = api
    monkeypatch.setattr("gated_lora.training.gated_trainer.time.sleep", lambda s: None)
    assert p.commit_final(["final_model"], [out / TRAINING_DONE], budget_s=3600)
    p.shutdown()
    assert len(api.commits) == 1


def test_no_commit_without_local_training_done(tmp_path):
    out = _outputs(tmp_path)
    (out / TRAINING_DONE).unlink()
    p = _HubPusher(out, "repo", "runX", enabled=True)
    api = _FakeApi()
    p._api = api
    assert not p.commit_final(["final_model"], [out / TRAINING_DONE], budget_s=10)
    p.shutdown()
    assert api.commits == []


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


class _HttpErr(Exception):
    def __init__(self, status):
        super().__init__(f"{status} Client Error")
        self.response = _Resp(status)


@pytest.mark.parametrize("exc", [_HttpErr(401), _HttpErr(403),
                                 RuntimeError("Invalid user token. The token is invalid.")])
def test_auth_error_is_not_retried(tmp_path, monkeypatch, exc):
    """A revoked token used to be retried until the slice deadline (3.5 h lost per run)."""
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("must not sleep/retry on 401/403"))
    p = _HubPusher(tmp_path, "r/x", "run", True)
    calls = []

    def fail():
        calls.append(1)
        raise exc

    with pytest.raises(type(exc)):
        p._retry("x", fail, budget_s=3600)
    assert len(calls) == 1


def test_rate_limit_is_still_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = _HubPusher(tmp_path, "r/x", "run", True)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _HttpErr(429)

    p._retry("x", flaky, budget_s=3600)
    assert len(calls) == 3
