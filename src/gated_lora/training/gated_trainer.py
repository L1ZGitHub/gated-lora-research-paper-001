"""
GatedLoRATrainer - unified trainer for PEFT LoRA baselines and Gated LoRA v2
(docs/v2_contract.md).

Features:
1. Step budget (`max_steps`), gradient accumulation with leftovers carried
   across epoch boundaries, exact-sample resume (`samples_seen`).
2. Mixed precision: autocast fp16/bf16 + GradScaler (fp16), fp32 trainable
   params and AdamW — same path for baseline and gated models.
3. Evaluation: answer loss / answer token accuracy / teacher-forced exact match,
   per task and overall; best model selected on the "val" mean task answer loss.
4. Deadline-aware stopping (env GLR_DEADLINE, fallback max_runtime_seconds) for
   SLURM job chaining; evals / routing analysis guarded by the same rule.
5. Atomic local checkpoints (`latest/`, `best_model/`, `final_model/`) and
   background HF Hub pushes (final push blocking, TRAINING_DONE verified).
6. Periodic routing analysis (gated models): per-layer, per-task usage averaged
   over real tokens, on a task-balanced loader.
"""

import concurrent.futures
import contextlib
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

_EVAL_CE_CHUNK = 2048  # rows per fp32 cross-entropy chunk in evaluate()

# Fallback runtime budget when GLR_DEADLINE is unset: 3h30 (4h SLURM limit).
# Override per-job with env GLR_MAX_RUNTIME_SECONDS (set by train.sbatch).
DEFAULT_MAX_RUNTIME_SECONDS = float(
    os.environ.get("GLR_MAX_RUNTIME_SECONDS", 3.5 * 3600)
)

TRAINING_DONE = "TRAINING_DONE"


def setup_logging_to_stdout():
    """Configure logging to output to stdout instead of stderr."""
    root_logger = logging.getLogger()

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    root_logger.setLevel(logging.INFO)


@dataclass
class TrainingState:
    """Tracks training progress (serialised to latest/training_state.json)."""
    global_step: int = 0  # optimizer steps taken
    epoch: int = 0  # current data epoch
    batch_idx: int = 0  # micro-batches consumed in the current epoch (informational)
    samples_seen: int = 0  # samples consumed in the current epoch (= resume skip_samples)
    total_samples_seen: int = 0
    best_eval_loss: float = float("inf")  # "val" mean task answer loss
    best_eval_step: int = -1
    last_eval_step: int = -1
    total_train_loss: float = 0.0
    total_lb_loss: float = 0.0  # Load balancing loss
    num_train_steps: int = 0  # micro-batches accounted in total_train_loss


@dataclass
class RoutingSnapshot:
    """Snapshot of routing patterns at a given step."""
    step: int
    epoch: int
    layer_expert_usage: List[List[float]]  # [num_layers, num_experts]
    task_layer_expert_usage: Dict[str, List[List[float]]]  # {task: [num_layers, num_experts]}
    layer_entropy: List[float]  # [num_layers]
    specialization_scores: Dict[str, float]  # per-layer specialization scores
    observed_layers: List[int] = field(default_factory=list)  # layers with real (gated) routing


# =============================================================================
# Small helpers
# =============================================================================

def _out(outputs: Any, key: str) -> Any:
    """Read a field from a dict output (gated) or a ModelOutput (PEFT)."""
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


def _make_grad_scaler(enabled: bool):
    try:
        from torch.amp import GradScaler
        return GradScaler("cuda", enabled=enabled)
    except (ImportError, TypeError):
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=enabled)


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": [random.getstate()[0], list(random.getstate()[1]), random.getstate()[2]],
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    if NUMPY_AVAILABLE:
        name, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = [name, keys.tolist(), int(pos), int(has_gauss), float(cached)]
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        v, internal, gauss = state["python"]
        random.setstate((v, tuple(internal), gauss))
    if "torch" in state:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            logger.warning("CUDA RNG state not restored: device count changed")
    if "numpy" in state and NUMPY_AVAILABLE:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class _HubPusher:
    """HF Hub uploads: one at a time on a background thread, blocking on demand.

    Uploads a hard-link snapshot of the directory so that a concurrent atomic
    re-save of e.g. best_model/ cannot change files under the uploader.
    Repo layout matches scripts/transfer/ensimag_push.py: <repo>/<run>/<dir>/.
    """

    RETRIES = 4
    # HF caps commits per repo (128/hour, shared by all runs): on a 429, wait for the window
    RATE_LIMIT_DELAY = 180.0

    def __init__(self, output_dir: Path, repo: str, run_name: str, enabled: bool):
        self.output_dir = output_dir
        self.repo = repo
        self.run_name = run_name
        self.enabled = enabled
        self._pending: Dict[str, concurrent.futures.Future] = {}
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-push")
            if enabled else None
        )
        self._api = None
        # Max seconds a push may spend retrying (None: RETRIES attempts). Set by the trainer
        # before deadline pushes so that a rate-limited push never outlives the job.
        self.budget_s: Optional[float] = None

    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=os.environ.get("HF_TOKEN"))
        return self._api

    def _retry(self, what: str, fn: Callable[[], Any], budget_s: Optional[float] = None) -> None:
        """Retry with backoff. With a time budget (arg, else self.budget_s), retry until the
        budget is spent instead of RETRIES times. A 429 (commit rate limit) waits at least
        RATE_LIMIT_DELAY."""
        budget = budget_s if budget_s is not None else self.budget_s
        t0 = time.time()
        delay = 15.0
        attempt = 0
        while True:
            attempt += 1
            try:
                fn()
                return
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate limit" in msg.lower():
                    delay = max(delay, self.RATE_LIMIT_DELAY)
                if budget is None:
                    give_up = attempt >= self.RETRIES
                else:
                    give_up = time.time() - t0 + delay > budget
                if give_up:
                    raise
                logger.warning(f"[hf-push] {what} failed (attempt {attempt}): "
                               f"{type(exc).__name__}: {msg.splitlines()[0][:200] if msg else ''}; "
                               f"retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 900.0)

    def _upload_snapshot(self, snap: Path, name: str) -> bool:
        path_in_repo = f"{self.run_name}/{name}"
        try:
            self._retry(path_in_repo, lambda: self.api().upload_folder(
                folder_path=str(snap),
                repo_id=self.repo,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                commit_message=f"Sync {path_in_repo}",
            ))
            logger.info(f"[hf-push] uploaded {path_in_repo}")
            return True
        except Exception:
            logger.exception(f"[hf-push] FAILED to upload {path_in_repo} (training continues)")
            return False
        finally:
            shutil.rmtree(snap, ignore_errors=True)

    def submit_dir(self, name: str, blocking: bool = False) -> bool:
        """Push output_dir/<name>/. Non-blocking unless `blocking`. Returns False if
        skipped (disabled / missing / same dir still uploading and not blocking)."""
        if not self.enabled:
            return False
        src = self.output_dir / name
        if not src.is_dir():
            return False
        prev = self._pending.get(name)
        if prev is not None and not prev.done():
            if not blocking:
                logger.info(f"[hf-push] {name} still uploading, skip this push")
                return False
            prev.result()
        snap = self.output_dir / ".push" / f"{name}-{time.time_ns()}"
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, snap, copy_function=_link_or_copy)
        fut = self._executor.submit(self._upload_snapshot, snap, name)
        self._pending[name] = fut
        if blocking:
            return bool(fut.result())
        return True

    def upload_file(self, local: Path, verify: bool = False) -> bool:
        """Blocking single-file upload (+ optional existence check)."""
        if not self.enabled or not local.is_file():
            return False
        path_in_repo = f"{self.run_name}/{local.name}"

        def _do():
            self.api().upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=self.repo,
                repo_type="dataset",
                commit_message=f"Sync {path_in_repo}",
            )
            if verify and not self.api().file_exists(self.repo, path_in_repo, repo_type="dataset"):
                raise RuntimeError(f"{path_in_repo} not visible on the Hub after upload")

        try:
            self._retry(path_in_repo, _do)
            logger.info(f"[hf-push] uploaded {path_in_repo}")
            return True
        except Exception:
            logger.exception(f"[hf-push] FAILED to upload {path_in_repo}")
            return False

    def commit_final(self, dirs: List[str], files: List[Path], budget_s: float) -> bool:
        """ONE atomic commit with every final artefact (dirs + root files; TRAINING_DONE must be
        among `files`), retried within `budget_s`, then TRAINING_DONE existence check.
        One commit instead of ~12 per run: HF allows 128 commits/hour per repo."""
        if not self.enabled:
            return False
        from huggingface_hub import CommitOperationAdd

        self.wait()
        snap_root = self.output_dir / ".push" / f"final-{time.time_ns()}"
        try:
            ops = []
            for name in dirs:
                src = self.output_dir / name
                if not src.is_dir():
                    continue
                snap = snap_root / name
                shutil.copytree(src, snap, copy_function=_link_or_copy)
                for f in sorted(snap.rglob("*")):
                    if f.is_file():
                        ops.append(CommitOperationAdd(
                            path_in_repo=f"{self.run_name}/{name}/{f.relative_to(snap).as_posix()}",
                            path_or_fileobj=str(f)))
            for f in files:
                if f.is_file():
                    ops.append(CommitOperationAdd(path_in_repo=f"{self.run_name}/{f.name}",
                                                  path_or_fileobj=str(f)))
            done_path = f"{self.run_name}/{TRAINING_DONE}"
            if not any(op.path_in_repo == done_path for op in ops):
                raise RuntimeError(f"{TRAINING_DONE} missing locally: not committing")

            def _do():
                self.api().create_commit(repo_id=self.repo, repo_type="dataset", operations=ops,
                                         commit_message=f"Final {self.run_name}")
                if not self.api().file_exists(self.repo, done_path, repo_type="dataset"):
                    raise RuntimeError(f"{done_path} not visible on the Hub after the commit")

            self._retry(f"final commit {self.run_name} ({len(ops)} files)", _do, budget_s=budget_s)
            logger.info(f"[hf-push] final commit {self.run_name}: {len(ops)} files incl. {TRAINING_DONE}")
            return True
        except Exception:
            logger.exception(f"[hf-push] FAILED final commit {self.run_name}")
            return False
        finally:
            shutil.rmtree(snap_root, ignore_errors=True)

    def wait(self) -> None:
        for fut in list(self._pending.values()):
            fut.result()

    def shutdown(self) -> None:
        if self._executor is not None:
            self.wait()
            self._executor.shutdown(wait=True)
            self._executor = None
            self.enabled = False


# =============================================================================
# Trainer
# =============================================================================

class GatedLoRATrainer:
    """
    Trainer for Gated LoRA v2 and PEFT LoRA baselines with:
    - step budget, grad accumulation carried across epochs, exact resume
    - mixed precision (autocast + GradScaler for fp16)
    - answer-level evaluation, best model on val mean task answer loss
    - deadline-aware stopping + atomic checkpoints + HF push for SLURM chaining
    - periodic routing analysis (gated models)
    """

    def __init__(
        self,
        model: nn.Module,
        data: Any = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
        output_dir: str = "./outputs",
        device: str = "cuda",
        max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,
        max_steps: Optional[int] = None,
        precision: Optional[str] = None,
    ):
        from gated_lora.training.config import resolve_precision

        self.model = model
        self.data = data  # MultiTaskDatasetLoader (DATA owner)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_runtime_seconds = max_runtime_seconds

        self.state = TrainingState()

        tc = config.training if config else None

        def _t(name, default):
            return getattr(tc, name, default) if tc is not None else default

        self.num_epochs = _t("num_epochs", 4)
        self.batch_size = _t("batch_size", 4)
        self.max_steps = int(max_steps if max_steps is not None else _t("max_steps", -1))
        if self.max_steps <= 0:
            raise ValueError("GatedLoRATrainer needs max_steps > 0 (pipeline.compute_max_steps)")
        self.seed = config.run_seed if config is not None else 42
        self.log_routing_stats = _t("log_routing_stats", True)
        self.collect_routing_stats = _t("collect_routing_stats", False)
        self.logging_steps = _t("logging_steps", 10)
        self.eval_steps = _t("eval_steps", 1000)
        self.eval_batch_size = _t("eval_batch_size", 32)
        self.eval_samples_per_task = _t("eval_samples_per_task_during_training", 250)
        self.save_steps = _t("save_steps", 500)
        self.save_margin = float(_t("save_margin_seconds", 300.0))
        self.gradient_accumulation_steps = _t("gradient_accumulation_steps", 1)
        self.max_grad_norm = _t("max_grad_norm", 1.0)
        self.routing_analysis_steps = _t("routing_analysis_steps", 500)
        self.routing_analysis_samples_per_task = _t("routing_analysis_samples_per_task", 32)
        # Exact only for right-padded batches (our collator): plain causal attention, no 4D mask
        self.skip_base_attention_mask = bool(_t("skip_base_attention_mask", False))
        self.generation_tasks = list(_t("generation_tasks", []) or [])
        self.generation_samples_per_task = int(_t("generation_samples_per_task", 500))
        self.generation_batch_size = int(_t("generation_batch_size", 32))
        dc = config.data if config else None
        self.sort_window = (self.batch_size * self.gradient_accumulation_steps
                            if dc is not None and getattr(dc, "sort_within_step", False) else 0)
        # Test hook (validate_v2.sbatch): stop like a deadline once global_step reaches it
        self.stop_at_step = int(os.environ.get("GLR_STOP_AT_STEP", "0") or 0)
        self.gate_reg_warmup_steps = (
            getattr(config.model, "gate_reg_warmup_ratio", 0.0) * self.max_steps if config else 0.0
        )
        if _t("gating_warmup_steps", 0) > 0 or _t("gating_warmup_epochs", 0) > 0:
            raise ValueError("Gating warmup was removed in v2 (experts are zero-initialised)")

        # Is the model gated? (PeftModel forwards unknown attrs to the base model,
        # which has no gating_network, so baselines resolve to False.)
        self.is_gated = hasattr(model, "gating_network") or bool(getattr(model, "is_gated", False))

        # Mixed precision
        self.precision = precision or resolve_precision(
            getattr(config.model, "precision", "auto") if config else "auto"
        )
        self._on_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.autocast_dtype = (
            {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
            if self._on_cuda else None
        )
        self.scaler = _make_grad_scaler(enabled=self.precision == "fp16" and self._on_cuda)
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        # Project only supervised positions to the vocabulary (see models/selective_head.py)
        from gated_lora.models.selective_head import install_selective_head
        try:
            hf_model = model.model if self.is_gated else model.get_base_model()
            self._head = install_selective_head(hf_model)
        except (AttributeError, ValueError) as e:  # e.g. test doubles without an HF head
            logger.warning(f"Selective output head not installed ({e}): full-sequence logits")
            self._head = None
        non_fp32 = [p.dtype for p in self.trainable_params if p.dtype != torch.float32]
        if non_fp32:
            raise ValueError(f"{len(non_fp32)} trainable params are not fp32 ({set(non_fp32)}); "
                             f"AdamW must run on fp32 params")

        # Routing statistics accumulator (detached GPU tensors; synced at logging)
        self.routing_stats_buffer: List[Dict[str, torch.Tensor]] = []
        self.routing_history: List[RoutingSnapshot] = []
        self.routing_analysis_failures = 0
        self.analysis_dataloader = None  # lazily built from DATA's balanced loader

        # Eval loaders cache + timing estimates
        self._eval_loaders: Dict[Tuple[str, Optional[int]], DataLoader] = {}
        self._eval_sec_per_batch: Optional[float] = None
        self.max_step_time = 0.0
        self._first_step_time: Optional[float] = None
        self.deadline: Optional[float] = None
        self._term_requested = False
        self._tok_real = 0
        self._tok_total = 0

        # HF push (only when configured AND HF_TOKEN present)
        hub_repo = os.environ.get("GLR_HF_REPO") or _t("hub_repo", "Helain/gated-lora-experiments")
        run_name = os.environ.get("GLR_RUN_NAME") or self.output_dir.name
        push_enabled = bool(_t("push_to_hub", True)) and bool(os.environ.get("HF_TOKEN"))
        if _t("push_to_hub", True) and not push_enabled:
            logger.warning("HF_TOKEN unset: HF Hub push disabled")
        self.pusher = _HubPusher(self.output_dir, hub_repo, run_name, push_enabled)
        self._best_dirty = False
        self._last_best_push_step: Optional[int] = None

        logger.info("GatedLoRATrainer initialized:")
        logger.info(f"  - Is gated model: {self.is_gated}")
        logger.info(f"  - Budget: {self.max_steps} optimizer steps "
                    f"(batch {self.batch_size} x accum {self.gradient_accumulation_steps})")
        logger.info(f"  - Precision: {self.precision} (autocast={self.autocast_dtype}, "
                    f"grad scaler={self.scaler.is_enabled()})")
        logger.info(f"  - Eval every {self.eval_steps} steps on {self.eval_samples_per_task}/task "
                    f"(batch {self.eval_batch_size})")
        logger.info(f"  - Routing analysis every: {self.routing_analysis_steps} steps")
        logger.info(f"  - HF push: {push_enabled} ({hub_repo}/{run_name})")

    # ------------------------------------------------------------------ utils

    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def _set_model_attr(self, name: str, value: Any) -> None:
        """Set a MODEL-contract attribute (reg_scale, collect_routing_stats) if present."""
        if self.is_gated and hasattr(self.model, name):
            setattr(self.model, name, value)

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device, non_blocking=True)

    def _forward(self, batch: Dict[str, Any], with_labels: bool, routing_info: bool = False):
        kwargs: Dict[str, Any] = {
            "input_ids": self._to_device(batch["input_ids"]),
            "attention_mask": self._to_device(batch["attention_mask"]),
        }
        if with_labels:
            kwargs["labels"] = self._to_device(batch["labels"])
        if self.is_gated:
            if routing_info:
                kwargs["return_routing_info"] = True
            if self.skip_base_attention_mask:
                kwargs["base_attention_mask"] = False  # hooks still get the mask
        else:
            kwargs["use_cache"] = False
            if self.skip_base_attention_mask:
                del kwargs["attention_mask"]
        return self.model(**kwargs)

    @contextlib.contextmanager
    def logits_off(self):
        """Forwards inside project ONE position to the vocabulary (logits unused: routing)."""
        if self._head is None:
            yield
            return
        self._head.index = torch.zeros(1, dtype=torch.long, device=self.device)
        try:
            yield
        finally:
            self._head.index = None

    @staticmethod
    def _supervised_positions(batch: Dict[str, Any], answer_only: bool):
        """Flat [B*T] indices of positions whose NEXT token is supervised, their targets and rows.

        Computed on the collate (CPU) tensors: no GPU sync.
        """
        key = "ans" if answer_only else "sup"
        if f"{key}_flat" in batch:  # computed (and pinned) by the DATA collator
            return batch[f"{key}_flat"], batch[f"{key}_tgt"], batch[f"{key}_rows"]
        from gated_lora.data.multi_task_dataset import supervised_positions
        return supervised_positions(batch["labels"],
                                    batch.get("answer_mask") if answer_only else None)

    def _forward_selected(self, batch: Dict[str, Any], answer_only: bool, routing_info: bool = False):
        """Forward without labels; logits [N, V] only at the supervised positions."""
        flat, tgt, rows = self._supervised_positions(batch, answer_only)
        flat = self._to_device(flat)
        if self._head is None:
            outputs = self._forward(batch, with_labels=False, routing_info=routing_info)
            logits = _out(outputs, "logits")
            logits = logits.reshape(-1, logits.shape[-1]).index_select(0, flat)
        else:
            self._head.index = flat
            try:
                outputs = self._forward(batch, with_labels=False, routing_info=routing_info)
            finally:
                self._head.index = None
            logits = _out(outputs, "logits")
        return outputs, logits, self._to_device(tgt), self._to_device(rows)

    # --------------------------------------------------------------- deadline

    def _init_deadline(self) -> None:
        env = os.environ.get("GLR_DEADLINE")
        if env:
            self.deadline = float(env)
            logger.info(f"Deadline from GLR_DEADLINE: "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.deadline))} "
                        f"({(self.deadline - time.time()) / 60:.1f} min left)")
        else:
            self.deadline = time.time() + self.max_runtime_seconds
            logger.info(f"GLR_DEADLINE unset: budget max_runtime_seconds="
                        f"{self.max_runtime_seconds / 3600:.2f} h")

    def _step_time_bound(self) -> float:
        if self.max_step_time > 0:
            return self.max_step_time
        return self._first_step_time or 0.0

    def time_allows(self, extra_seconds: float = 0.0) -> bool:
        """True if `extra_seconds` of work still leaves 2 steps + save margin before the deadline."""
        if self.deadline is None:
            return True
        return time.time() + extra_seconds + 2 * self._step_time_bound() + self.save_margin <= self.deadline

    def _eval_seconds_estimate(self, loader: DataLoader) -> float:
        try:
            n = len(loader)
        except TypeError:
            return 0.0
        per = self._eval_sec_per_batch
        if per is None:
            per = self._step_time_bound() / max(self.gradient_accumulation_steps, 1)
        return n * per

    def estimate_routing_analysis_seconds(self) -> float:
        try:
            return self._eval_seconds_estimate(self.get_analysis_dataloader())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ data

    def _train_loader(self) -> DataLoader:
        return self.data.create_weighted_dataloader(
            split="train",
            batch_size=self.batch_size,
            seed=self.seed,
            epoch=self.state.epoch,
            skip_samples=self.state.samples_seen,
            sort_window=self.sort_window,
        )

    def _eval_loader(self, which: str, samples_per_task: Optional[int]) -> DataLoader:
        key = (which, samples_per_task)
        if key not in self._eval_loaders:
            self._eval_loaders[key] = self.data.create_eval_dataloader(
                which=which, batch_size=self.eval_batch_size, samples_per_task=samples_per_task,
            )
        return self._eval_loaders[key]

    def set_analysis_dataloader(self, dataloader: DataLoader):
        """Override the dataloader used for routing analysis (default: DATA balanced loader)."""
        self.analysis_dataloader = dataloader
        logger.info(f"Analysis dataloader set with {len(dataloader)} batches")

    def get_analysis_dataloader(self) -> DataLoader:
        if self.analysis_dataloader is None:
            self.analysis_dataloader = self.data.create_balanced_loader(
                samples_per_task=self.routing_analysis_samples_per_task,
                batch_size=self.eval_batch_size,
            )
        return self.analysis_dataloader

    # --------------------------------------------------------- routing stats

    def _extract_routing_stats(self, outputs: Any) -> Dict[str, torch.Tensor]:
        """Detached routing statistics (no host sync; converted at logging time)."""
        stats: Dict[str, torch.Tensor] = {}
        routing_info = _out(outputs, "routing_info") or {}
        for key, name in (("mean_entropy", "routing_entropy"),
                          ("mean_top1_dominance", "top1_dominance")):
            v = routing_info.get(key)
            if isinstance(v, torch.Tensor):
                stats[name] = v.detach().float().reshape(())
            elif v is not None:
                stats[name] = torch.tensor(float(v))
        usage = routing_info.get("mean_expert_usage")
        if isinstance(usage, torch.Tensor):
            usage = usage.detach().float().reshape(-1)
            for i, u in enumerate(usage):
                stats[f"expert_{i}_usage"] = u
            # load imbalance = std of expert usage across experts
            stats["load_imbalance"] = usage.std(unbiased=False)
        return stats

    def _aggregate_routing_stats(self) -> Dict[str, float]:
        """Aggregate buffered routing statistics (single host sync)."""
        if not self.routing_stats_buffer:
            return {}
        keys = list(self.routing_stats_buffer[0].keys())
        aggregated = {}
        for key in keys:
            values = [s[key].to(self.device) for s in self.routing_stats_buffer if key in s]
            if values:
                aggregated[f"avg_{key}"] = torch.stack(values).mean()
        self.routing_stats_buffer.clear()
        names = list(aggregated)
        vals = torch.stack([aggregated[k] for k in names]).tolist() if names else []
        return dict(zip(names, vals))

    def run_routing_analysis(self, dataloader: Optional[DataLoader] = None) -> Optional[RoutingSnapshot]:
        """
        Per-layer, per-task routing snapshot on a task-balanced loader.

        gate_weights are [batch, seq, experts]: usage = mean over (batch, real
        tokens) using attention_mask; per task = mean over that task's real tokens.
        Layers reporting uniform (non-gated) routing are excluded from
        `observed_layers` and reported as uniform.
        """
        if not self.is_gated:
            return None
        loader = dataloader or self.get_analysis_dataloader()

        num_experts = int(getattr(self.model, "num_experts"))
        num_layers = int(getattr(self.model, "num_layers"))

        # per layer: list of ([B, E] weight sums over real tokens, [B] real-token counts)
        per_layer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        row_tasks: List[str] = []
        prev_collect = getattr(self.model, "collect_routing_stats", None)
        self._set_model_attr("collect_routing_stats", True)
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                for batch in loader:
                    with self.logits_off():
                        outputs = self._forward(batch, with_labels=False, routing_info=True)
                    mask = self._to_device(batch["attention_mask"]).float()
                    tasks = list(batch.get("task") or ["all"] * mask.size(0))
                    row_tasks.extend(tasks)
                    pli = (_out(outputs, "routing_info") or {}).get("per_layer_info", {})
                    for layer_key, info in pli.items():
                        if info.get("uniform") or "gate_weights" not in info:
                            continue
                        layer_idx = int(layer_key)
                        gw = info["gate_weights"].float()
                        if gw.dim() == 2:  # [B, E] (pooled) -> one "token" per row
                            gw, m = gw[:, None, :], torch.ones_like(mask[:, :1])
                        else:
                            m = mask
                        if gw.shape[:2] != m.shape:
                            raise ValueError(f"layer {layer_idx}: gate_weights {tuple(gw.shape)} "
                                             f"vs attention_mask {tuple(m.shape)}")
                        wsum = (gw * m.unsqueeze(-1)).sum(dim=1)  # [B, E]
                        per_layer.setdefault(layer_idx, []).append((wsum, m.sum(dim=1)))
                    # A layer missing from some batch would misalign rows -> fail loudly.
                    for layer_idx, chunks in per_layer.items():
                        if sum(c[1].numel() for c in chunks) != len(row_tasks):
                            raise ValueError(f"layer {layer_idx} missing routing info in some batch")
        finally:
            self.model.train()
            if prev_collect is not None:
                self._set_model_attr("collect_routing_stats", prev_collect)

        uniform = [1.0 / num_experts] * num_experts
        layer_usage = [list(uniform) for _ in range(num_layers)]
        task_names = sorted(set(row_tasks))
        task_usage: Dict[str, List[List[float]]] = {t: [list(uniform) for _ in range(num_layers)]
                                                    for t in task_names}
        observed = sorted(per_layer)
        for layer_idx in observed:
            W = torch.cat([c[0] for c in per_layer[layer_idx]]).cpu()  # [N, E]
            C = torch.cat([c[1] for c in per_layer[layer_idx]]).cpu()  # [N]
            total = C.sum().clamp(min=1.0)
            layer_usage[layer_idx] = (W.sum(0) / total).tolist()
            for t in task_names:
                rows = torch.tensor([rt == t for rt in row_tasks])
                ct = C[rows].sum()
                if ct > 0:
                    task_usage[t][layer_idx] = (W[rows].sum(0) / ct).tolist()

        layer_entropy = []
        for layer_idx in range(num_layers):
            if layer_idx not in per_layer:
                layer_entropy.append(0.0)
                continue
            layer_entropy.append(-sum(p * math.log(p) for p in layer_usage[layer_idx] if p > 0))

        # Specialization per layer: mean over experts of the across-task variance
        specialization_scores = {}
        for layer_idx in range(num_layers):
            score = 0.0
            if layer_idx in per_layer and len(task_names) >= 2:
                variances = []
                for e in range(num_experts):
                    vals = [task_usage[t][layer_idx][e] for t in task_names]
                    mu = sum(vals) / len(vals)
                    variances.append(sum((v - mu) ** 2 for v in vals) / len(vals))
                score = sum(variances) / len(variances)
            specialization_scores[f"layer_{layer_idx}"] = score

        return RoutingSnapshot(
            step=self.state.global_step,
            epoch=self.state.epoch,
            layer_expert_usage=layer_usage,
            task_layer_expert_usage=task_usage,
            layer_entropy=layer_entropy,
            specialization_scores=specialization_scores,
            observed_layers=observed,
        )

    def _routing_analysis_safely(self) -> None:
        """Periodic routing analysis: never crashes training, errors stay visible."""
        if not self.time_allows(self.estimate_routing_analysis_seconds()):
            logger.warning(f"Routing analysis at step {self.state.global_step} skipped: deadline")
            return
        logger.info(f"Running routing analysis at step {self.state.global_step}...")
        try:
            snapshot = self.run_routing_analysis()
        except Exception:
            self.routing_analysis_failures += 1
            logger.exception(f"Routing analysis FAILED at step {self.state.global_step} "
                             f"(failure #{self.routing_analysis_failures}; training continues)")
            return
        if snapshot is None:
            return
        self.routing_history.append(snapshot)
        obs = snapshot.observed_layers
        if not obs:
            logger.warning("Routing analysis: no gated layer reported gate_weights")
            return
        spec = {k: v for k, v in snapshot.specialization_scores.items() if int(k.split("_")[1]) in obs}
        max_layer, max_score = max(spec.items(), key=lambda x: x[1])
        mean_entropy = sum(snapshot.layer_entropy[i] for i in obs) / len(obs)
        logger.info(f"  Max specialization: {max_layer} = {max_score:.4f}, "
                    f"mean usage entropy = {mean_entropy:.4f}")
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "routing/max_specialization_score": max_score,
                "routing/max_specialization_layer": int(max_layer.split("_")[1]),
                "routing/mean_entropy": mean_entropy,
            }, step=self.state.global_step)

    def _save_routing_history(self, directory: Optional[Path] = None):
        """Save routing history to file."""
        if not self.routing_history:
            return
        history_path = (directory or self.output_dir) / "routing_history.json"
        history_data = [asdict(snapshot) for snapshot in self.routing_history]
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        logger.info(f"Saved {len(history_data)} routing snapshots to {history_path}")

    # -------------------------------------------------------------- training

    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward + backward on one micro-batch.

        Returns detached [loss, lm_loss, load_balancing_loss, entropy_loss] on device
        (no host sync).
        """
        self.model.train()
        want_routing = self.is_gated and self.collect_routing_stats and self.log_routing_stats
        with self._autocast():
            outputs, logits, tgt, _ = self._forward_selected(batch, answer_only=False,
                                                             routing_info=want_routing)
            # Drop every reference to the fp16 [N, V] logits before backward. Gated: plain dict
            # (routing stats may still be read); baseline: HF ModelOutput (no pop), not needed.
            if self.is_gated:
                outputs.pop("logits", None)
            else:
                outputs = None
            # Mean over supervised tokens (= HF causal-LM loss). Every example has >= 1 (EOS).
            lm_loss = F.cross_entropy(logits.float(), tgt)
            loss = lm_loss
            aux: Dict[str, torch.Tensor] = {}
            if self.is_gated:
                aux = self.model.aux_losses(lm_loss.device)
                loss = loss + aux.pop("total")
        del logits
        scaled_loss = loss.float() / self.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        if want_routing:
            routing_stats = self._extract_routing_stats(outputs)
            if routing_stats:
                self.routing_stats_buffer.append(routing_stats)

        loss_d = loss.detach().float().reshape(())
        zero = torch.zeros((), device=loss_d.device)
        return torch.stack([
            loss_d,
            lm_loss.detach().float().reshape(()),
            aux.get("load_balancing_loss", zero).detach().float().reshape(()),
            aux.get("l1_gate_loss", zero).detach().float().reshape(()),
        ])

    def _optimizer_step(self) -> torch.Tensor:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm.detach()

    def _update_reg_scale(self) -> None:
        if self.gate_reg_warmup_steps > 0:
            scale = min(1.0, self.state.global_step / self.gate_reg_warmup_steps)
        else:
            scale = 1.0
        self._set_model_attr("reg_scale", float(scale))

    def _log_train(self, loss_sum: torch.Tensor, n_micro: int, grad_norm: Optional[torch.Tensor],
                   progress_bar: Any) -> None:
        if n_micro == 0:
            return
        packed = torch.cat([loss_sum / n_micro,
                            (grad_norm if grad_norm is not None else torch.zeros((), device=loss_sum.device))
                            .float().reshape(1)])
        loss, lm_loss, lb_loss, ent_loss, gnorm = packed.tolist()  # the only host sync
        self.state.total_train_loss += loss * n_micro
        self.state.total_lb_loss += lb_loss * n_micro
        self.state.num_train_steps += n_micro
        lr = self.optimizer.param_groups[0]["lr"]
        remaining = (self.deadline - time.time()) if self.deadline else float("nan")
        log_dict = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "loss": loss,
            "lm_loss": lm_loss,
            "load_balancing_loss": lb_loss,
            "entropy_reg_loss": ent_loss,
            "grad_norm": gnorm,
            "learning_rate": lr,
            "max_step_seconds": self.max_step_time,
            "remaining_minutes": remaining / 60,
            "padding_efficiency": self._tok_real / max(self._tok_total, 1),
        }
        self._tok_real = self._tok_total = 0
        if self.scaler.is_enabled():
            log_dict["grad_scale"] = self.scaler.get_scale()
        # fp16: an isolated non-finite step is skipped by the scaler; only a collapsed scale
        # (>= 16 overflows in a row from 65536) means the run is broken.
        if (not self.scaler.is_enabled() and not math.isfinite(lm_loss)) \
                or log_dict.get("grad_scale", 1.0) < 1.0:
            raise FloatingPointError(
                f"step {self.state.global_step}: lm_loss={lm_loss}, "
                f"grad_scale={log_dict.get('grad_scale')} -> the {self.precision} forward overflows "
                f"(e.g. Gemma-2 / Qwen2.5 in fp16); set model.precision to bf16 (A40) or fp32")
        if self.is_gated:
            log_dict["reg_scale"] = float(getattr(self.model, "reg_scale", 1.0))
        log_dict.update(self._aggregate_routing_stats())

        if TQDM_AVAILABLE and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}",
                                      "remain": f"{remaining / 60:.0f}m"})
        logger.info(
            f"Step {self.state.global_step}/{self.max_steps} (epoch {self.state.epoch}, "
            f"sample {self.state.samples_seen}): loss={loss:.4f} lm={lm_loss:.4f} "
            f"lb={lb_loss:.4f} ent={ent_loss:.4f} gnorm={gnorm:.3f} lr={lr:.2e} "
            f"pad_eff={log_dict['padding_efficiency']:.2f} "
            f"step_max={self.max_step_time:.2f}s remaining={remaining / 60:.1f}min"
        )
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(log_dict, step=self.state.global_step)

    # ------------------------------------------------------------ evaluation

    def generation_eval(self, which: str = "final") -> Dict[str, Any]:
        """Greedy generation on the first `generation_samples_per_task` examples of each task in
        `generation_tasks` (left-padded, KV cache), scored by training/generation_eval.py.
        Returns {task: {metric, mean, num_examples, per_example, predictions, example_idx}}."""
        from gated_lora.training.generation_eval import GEN_MAX_NEW_TOKENS, score

        tok = self.data.tokenizer
        pad_id, eos_id = self.data.pad_id, tok.eos_token_id
        self.model.eval()
        results: Dict[str, Any] = {}
        try:
            for task in self.generation_tasks:
                t0 = time.time()
                exs = self.data.generation_examples(task, which, self.generation_samples_per_task)
                exs = sorted(exs, key=lambda e: len(e["prompt_ids"]))  # less padding
                max_new = GEN_MAX_NEW_TOKENS.get(task, 64)
                preds: List[str] = []
                for i in range(0, len(exs), self.generation_batch_size):
                    chunk = exs[i:i + self.generation_batch_size]
                    T = max(len(e["prompt_ids"]) for e in chunk)
                    ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
                    mask = torch.zeros((len(chunk), T), dtype=torch.long)
                    for r, e in enumerate(chunk):  # LEFT padding for generation
                        n = len(e["prompt_ids"])
                        ids[r, T - n:] = e["prompt_ids"]
                        mask[r, T - n:] = 1
                    with torch.no_grad(), self._autocast():
                        out = self.model.generate(
                            input_ids=self._to_device(ids), attention_mask=self._to_device(mask),
                            max_new_tokens=max_new, do_sample=False, num_beams=1,
                            pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True)
                    for row in out[:, T:].cpu():
                        preds.append(tok.decode(row, skip_special_tokens=True).strip())
                refs = [e["reference"] for e in exs]
                res = score(task, preds, refs)
                res.update(example_idx=[int(e["example_idx"]) for e in exs], predictions=preds,
                           seconds=time.time() - t0)
                results[task] = res
                logger.info(f"Generation eval {task} ({which}): {res['metric']}={res['mean']:.4f} "
                            f"on {res['num_examples']} examples ({res['seconds']:.0f}s)")
        finally:
            self.model.train()
        return results

    def _generation_seconds_estimate(self) -> float:
        # ~0.05 s per generated token step per batch (hooks are Python), x tasks
        from gated_lora.training.generation_eval import GEN_MAX_NEW_TOKENS
        n_batches = math.ceil(self.generation_samples_per_task / self.generation_batch_size)
        return sum(n_batches * GEN_MAX_NEW_TOKENS.get(t, 64) * 0.05 + 60
                   for t in self.generation_tasks)

    def _accumulate_top1(self, outputs: Any, batch: Dict[str, Any],
                         acc: Dict[int, List[torch.Tensor]]) -> None:
        """Token-level top-1 gate weight per layer (same statistic as the frozen-gate
        calibration), summed on device."""
        pli = (_out(outputs, "routing_info") or {}).get("per_layer_info", {})
        am = self._to_device(batch["attention_mask"]).float()
        n = am.sum()
        for layer_key, info in pli.items():
            gw = info.get("gate_weights")
            if info.get("uniform") or gw is None or gw.dim() != 3:
                continue
            top1 = (gw.float().max(dim=-1).values * am).sum()
            a = acc.setdefault(int(layer_key), [torch.zeros((), device=am.device),
                                                torch.zeros((), device=am.device)])
            a[0] += top1
            a[1] += n

    def _row_gate_means(self, outputs: Any, batch: Dict[str, Any]):
        """Per row: mean gate weights per layer over prompt tokens and over answer tokens
        ([B, L, E] each, NaN for layers without learned routing)."""
        pli = (_out(outputs, "routing_info") or {}).get("per_layer_info", {})
        am = self._to_device(batch["attention_mask"]).float()
        ans = self._to_device(batch["answer_mask"]).float() * am
        prm = am - ans
        L, E = self.model.num_layers, self.model.num_experts
        B = am.size(0)
        gp = torch.full((B, L, E), float("nan"), device=am.device)
        ga = torch.full((B, L, E), float("nan"), device=am.device)
        for layer_key, info in pli.items():
            gw = info.get("gate_weights")
            if info.get("uniform") or gw is None or gw.dim() != 3:
                continue
            gw = gw.float()
            l = int(layer_key)
            gp[:, l] = (gw * prm.unsqueeze(-1)).sum(1) / prm.sum(1, keepdim=True).clamp(min=1)
            ga[:, l] = (gw * ans.unsqueeze(-1)).sum(1) / ans.sum(1, keepdim=True).clamp(min=1)
        return gp.half(), ga.half()

    def _write_example_dump(self, path: Path, row_tasks: List[str], S: List[List[float]],
                            ex_meta: List[Tuple[torch.Tensor, torch.Tensor]],
                            ex_gates: List[Tuple[torch.Tensor, torch.Tensor]],
                            top1_acc: Optional[Dict[int, List[torch.Tensor]]] = None) -> None:
        """npz, one entry per example: task, example_idx (index in the task's tokenised final
        subset), prompt_len, answer_nll_sum, answer_tokens, answer_correct (EM = correct ==
        tokens), and gate_prompt / gate_answer [N, L, E] float16 (gated models) +
        layer_top1_dominance [L] (token-level mean of the max gate weight over all real tokens,
        NaN for layers without learned routing: the frozen-gate calibration target)."""
        arrays: Dict[str, Any] = {
            "task": np.array(row_tasks),
            "example_idx": torch.cat([m[0] for m in ex_meta]).numpy(),
            "prompt_len": torch.cat([m[1] for m in ex_meta]).numpy(),
            "answer_nll_sum": np.array([r[0] for r in S], dtype=np.float32),
            "answer_tokens": np.array([r[1] for r in S], dtype=np.int32),
            "answer_correct": np.array([r[2] for r in S], dtype=np.int32),
        }
        if top1_acc:
            top1 = np.full(self.model.num_layers, np.nan, dtype=np.float32)
            for l, (num, den) in top1_acc.items():
                top1[l] = float(num) / max(float(den), 1.0)
            arrays["layer_top1_dominance"] = top1
        if ex_gates:
            arrays["gate_prompt"] = torch.cat([g[0] for g in ex_gates]).cpu().numpy()
            arrays["gate_answer"] = torch.cat([g[1] for g in ex_gates]).cpu().numpy()
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        logger.info(f"Per-example dump: {len(row_tasks)} examples -> {path}")

    def evaluate(self, which: str = "val", samples_per_task: Optional[int] = None,
                 dump_path: Optional[Path] = None) -> Dict[str, Any]:
        """Answer-level evaluation (autocast + no_grad).

        Per task and overall: answer loss (mean NLL over answer tokens, i.e.
        labels != -100 and answer_mask), answer token accuracy, and
        teacher-forced exact match (every answer token incl. EOS argmax-correct).
        `eval_loss` = mean over tasks of the per-task answer loss.

        ``dump_path``: also write per-example arrays (npz, see `_write_example_dump`): task,
        example_idx, prompt_len, answer NLL sum / token count / correct count and, for gated
        models, the per-layer mean gate weights over PROMPT tokens and over ANSWER tokens.
        """
        loader = self._eval_loader(which, samples_per_task)
        prev_collect = getattr(self.model, "collect_routing_stats", None)
        self._set_model_attr("collect_routing_stats", False)
        self.model.eval()
        t0 = time.time()
        rows: List[torch.Tensor] = []  # [B, 3]: nll_sum, n_tok, n_correct
        row_tasks: List[str] = []
        ex_meta: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ex_gates: List[Tuple[torch.Tensor, torch.Tensor]] = []
        top1_acc: Dict[int, List[torch.Tensor]] = {}  # layer -> [sum(top1 * tokens), tokens]
        n_batches = 0
        try:
            with torch.no_grad(), self._autocast():
                for batch in loader:
                    want_gates = dump_path is not None and self.is_gated
                    outputs, sel, tgt, row_idx = self._forward_selected(
                        batch, answer_only=True, routing_info=want_gates)
                    if dump_path is not None:
                        ex_meta.append((batch["example_idx"], batch["prompt_len"]))
                        if want_gates:
                            ex_gates.append(self._row_gate_means(outputs, batch))
                            self._accumulate_top1(outputs, batch, top1_acc)
                    del outputs  # sel: [N, V] answer positions only
                    # Row chunks: an fp32 [N, V] copy (+ log-softmax) is ~30 GB for a
                    # 152k vocab at N ~ 14k answer tokens.
                    nll = torch.empty(tgt.shape[0], device=tgt.device, dtype=torch.float32)
                    correct = torch.empty_like(nll)
                    for s in range(0, tgt.shape[0], _EVAL_CE_CHUNK):
                        c = sel[s:s + _EVAL_CE_CHUNK].float()
                        t = tgt[s:s + _EVAL_CE_CHUNK]
                        nll[s:s + _EVAL_CE_CHUNK] = F.cross_entropy(c, t, reduction="none")
                        correct[s:s + _EVAL_CE_CHUNK] = (c.argmax(dim=-1) == t).float()
                    del sel
                    bsz = batch["labels"].size(0)
                    z = torch.zeros(bsz, device=tgt.device)
                    rows.append(torch.stack([
                        z.index_add(0, row_idx, nll),
                        z.index_add(0, row_idx, torch.ones_like(nll)),
                        z.index_add(0, row_idx, correct),
                    ], dim=1))
                    row_tasks.extend(batch["task"])
                    n_batches += 1
        finally:
            self.model.train()
            if prev_collect is not None:
                self._set_model_attr("collect_routing_stats", prev_collect)

        elapsed = time.time() - t0
        if n_batches:
            per_batch = elapsed / n_batches
            self._eval_sec_per_batch = max(self._eval_sec_per_batch or 0.0, per_batch)
        if not rows:
            return {"which": which, "num_examples": 0, "eval_seconds": elapsed}

        S = torch.cat(rows).cpu().tolist()  # single host sync
        if dump_path is not None:
            self._write_example_dump(dump_path, row_tasks, S, ex_meta, ex_gates, top1_acc)
        acc: Dict[str, List[float]] = {}
        for task, (nll_sum, ntok, ncorr) in zip(row_tasks, S):
            a = acc.setdefault(task, [0.0, 0.0, 0.0, 0, 0])
            a[0] += nll_sum
            a[1] += ntok
            a[2] += ncorr
            if ntok > 0:
                a[3] += 1
                a[4] += int(ncorr == ntok)

        per_task: Dict[str, Dict[str, float]] = {}
        for task, (nll_sum, ntok, ncorr, n_ex, n_em) in sorted(acc.items()):
            per_task[task] = {
                "answer_loss": nll_sum / ntok if ntok else float("nan"),
                "answer_token_acc": ncorr / ntok if ntok else float("nan"),
                "exact_match": n_em / n_ex if n_ex else float("nan"),
                "num_examples": n_ex,
                "num_answer_tokens": int(ntok),
            }
        tot = [sum(a[i] for a in acc.values()) for i in range(5)]
        valid = [m for m in per_task.values() if m["num_answer_tokens"] > 0]
        mean_task_loss = sum(m["answer_loss"] for m in valid) / len(valid) if valid else float("nan")
        result = {
            "which": which,
            "step": self.state.global_step,
            "samples_per_task": samples_per_task,
            "eval_loss": mean_task_loss,
            "mean_task_answer_loss": mean_task_loss,
            "mean_task_exact_match": (sum(m["exact_match"] for m in valid) / len(valid)) if valid else float("nan"),
            "answer_loss": tot[0] / tot[1] if tot[1] else float("nan"),
            "answer_token_acc": tot[2] / tot[1] if tot[1] else float("nan"),
            "exact_match": tot[4] / tot[3] if tot[3] else float("nan"),
            "num_examples": int(tot[3]),
            "eval_seconds": elapsed,
            "per_task": per_task,
        }
        return result

    def _log_eval(self, metrics: Dict[str, Any], tag: str) -> None:
        logger.info(
            f"[{tag}] step {self.state.global_step}: mean_task_answer_loss="
            f"{metrics.get('mean_task_answer_loss', float('nan')):.4f} "
            f"answer_loss={metrics.get('answer_loss', float('nan')):.4f} "
            f"tok_acc={metrics.get('answer_token_acc', float('nan')):.4f} "
            f"EM={metrics.get('exact_match', float('nan')):.4f} "
            f"({metrics.get('num_examples', 0)} ex, {metrics.get('eval_seconds', 0):.0f}s)"
        )
        for task, m in metrics.get("per_task", {}).items():
            logger.info(f"    {task:>14}: loss={m['answer_loss']:.4f} acc={m['answer_token_acc']:.4f} "
                        f"EM={m['exact_match']:.4f} (n={m['num_examples']})")
        if WANDB_AVAILABLE and wandb.run is not None:
            flat = {f"{tag}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            for task, m in metrics.get("per_task", {}).items():
                flat.update({f"{tag}/{task}/{k}": v for k, v in m.items()})
            wandb.log(flat, step=self.state.global_step)

    def _maybe_eval_for_selection(self) -> None:
        """Subset "val" eval for best-model selection (at most once per step)."""
        if self.state.last_eval_step == self.state.global_step:
            return
        loader = self._eval_loader("val", self.eval_samples_per_task)
        if not self.time_allows(self._eval_seconds_estimate(loader)):
            logger.warning(f"Eval at step {self.state.global_step} skipped: deadline too close")
            return
        metrics = self.evaluate("val", self.eval_samples_per_task)
        self.state.last_eval_step = self.state.global_step
        self._log_eval(metrics, "eval")
        loss = metrics.get("eval_loss", float("nan"))
        if loss == loss and loss < self.state.best_eval_loss:  # NaN-safe
            logger.info(f"  New best val mean task answer loss {loss:.4f} "
                        f"(prev {self.state.best_eval_loss:.4f})")
            self.state.best_eval_loss = loss
            self.state.best_eval_step = self.state.global_step
            self._save_weights_dir("best_model")
            self._best_dirty = True

    # ----------------------------------------------------------- checkpoints

    def _atomic_dir_save(self, name: str, writer: Callable[[Path], None]) -> Path:
        """Write <name>.tmp/ then swap it in place (old copy kept as <name>.old/ until done)."""
        final = self.output_dir / name
        tmp = self.output_dir / f"{name}.tmp"
        old = self.output_dir / f"{name}.old"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        writer(tmp)
        if final.exists():
            if old.exists():
                shutil.rmtree(old)
            os.replace(final, old)
        os.replace(tmp, final)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        return final

    def _write_weights(self, directory: Path) -> None:
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(directory))
        else:
            trainable = {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
            torch.save(trainable, directory / "model.pt")

    def _save_weights_dir(self, name: str) -> None:
        self._atomic_dir_save(name, self._write_weights)
        logger.info(f"Saved {name}/ (weights only)")

    def _training_state_dict(self) -> Dict[str, Any]:
        st = asdict(self.state)
        st.update({
            "max_steps": self.max_steps,
            "seed": self.seed,
            "precision": self.precision,
            "max_step_time": self.max_step_time,
            "routing_analysis_failures": self.routing_analysis_failures,
            "rng": _rng_state(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return st

    def save_checkpoint(self, name: str = "latest"):
        """Save the resumable `latest/` checkpoint (atomic, local only).

        Contents: adapter weights + optimizer.pt + scheduler.pt + scaler.pt +
        training_state.json (global_step, epoch, samples_seen, RNG states) +
        routing_history.json. Only valid at an optimizer-step boundary.
        """
        if name != "latest":
            raise ValueError("Only latest/ carries optimizer state in v2; use _save_weights_dir")

        def _writer(d: Path) -> None:
            self._write_weights(d)
            torch.save(self.optimizer.state_dict(), d / "optimizer.pt")
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), d / "scheduler.pt")
            if self.scaler.is_enabled():
                torch.save(self.scaler.state_dict(), d / "scaler.pt")
            with open(d / "training_state.json", "w") as f:
                json.dump(self._training_state_dict(), f)
            self._save_routing_history(d)

        self._atomic_dir_save("latest", _writer)
        logger.info(f"Saved latest/ at step {self.state.global_step} "
                    f"(epoch {self.state.epoch}, samples_seen {self.state.samples_seen})")

    def find_latest_checkpoint(self) -> Optional[Path]:
        """latest/ (or latest.old/ if a crash happened mid-swap) with a training_state.json."""
        for name in ("latest", "latest.old"):
            d = self.output_dir / name
            if (d / "training_state.json").exists():
                return d
        return None

    def load_checkpoint(self, path: str):
        """Load a checkpoint: weights (both model families), optimizer, scheduler,
        scaler, training state and RNG. Refuses to resume without weights."""
        checkpoint_dir = Path(path)

        # Load model state
        if (checkpoint_dir / "model.pt").exists():
            state_dict = torch.load(
                checkpoint_dir / "model.pt", map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model from {checkpoint_dir / 'model.pt'}")
        elif hasattr(self.model, "load_adapter_state"):
            # GatedLoRAModelV2: restore experts + gating in place
            self.model.load_adapter_state(str(checkpoint_dir))
        elif (checkpoint_dir / "adapter_model.safetensors").exists() or (
            checkpoint_dir / "adapter_model.bin"
        ).exists():
            # PEFT baseline LoRA: restore adapter weights in place
            from peft.utils import set_peft_model_state_dict

            st_path = checkpoint_dir / "adapter_model.safetensors"
            if st_path.exists():
                from safetensors.torch import load_file

                adapter_state = load_file(str(st_path), device=str(self.device))
            else:
                adapter_state = torch.load(
                    checkpoint_dir / "adapter_model.bin",
                    map_location=self.device,
                    weights_only=True,
                )
            set_peft_model_state_dict(self.model, adapter_state)
            logger.info(f"Loaded PEFT adapter weights from {checkpoint_dir}")
        else:
            raise FileNotFoundError(
                f"Cannot resume: no loadable model weights found in {checkpoint_dir} "
                f"(looked for model.pt / expert_pools.pt+gating_network.pt / adapter_model.*). "
                f"Resuming without weights would silently restart training."
            )
        bad = [p.dtype for p in self.trainable_params if p.dtype != torch.float32]
        if bad:
            raise RuntimeError(f"Trainable params lost fp32 after loading weights: {set(bad)}")

        if (checkpoint_dir / "optimizer.pt").exists():
            self.optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pt", map_location=self.device)
            )
            logger.info("Loaded optimizer state")
        else:
            logger.warning("No optimizer.pt in checkpoint: optimizer restarts from scratch")

        if self.scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
            self.scheduler.load_state_dict(
                torch.load(checkpoint_dir / "scheduler.pt", map_location=self.device)
            )
            logger.info("Loaded scheduler state")

        if self.scaler.is_enabled() and (checkpoint_dir / "scaler.pt").exists():
            self.scaler.load_state_dict(torch.load(checkpoint_dir / "scaler.pt"))
            logger.info("Loaded grad scaler state")

        if (checkpoint_dir / "training_state.json").exists():
            with open(checkpoint_dir / "training_state.json", "r") as f:
                st = json.load(f)
            for key in TrainingState.__dataclass_fields__:
                if key in st:
                    setattr(self.state, key, type(getattr(TrainingState(), key))(st[key]))
            if "samples_seen" not in st:
                # Legacy checkpoint: batch_idx was the next batch to process.
                self.state.samples_seen = int(st.get("batch_idx", 0)) * self.batch_size
                logger.warning(f"Legacy training_state.json (no samples_seen): resuming at "
                               f"sample {self.state.samples_seen} of epoch {self.state.epoch}")
            if st.get("precision") not in (None, self.precision):
                logger.warning(f"PRECISION CHANGED since checkpoint: {st.get('precision')} -> "
                               f"{self.precision} (e.g. after an fp16 overflow abort); resuming "
                               f"its weights and optimizer state anyway")
            if st.get("max_steps") not in (None, self.max_steps):
                logger.warning(f"max_steps changed since checkpoint: {st.get('max_steps')} -> {self.max_steps}")
            self.max_step_time = float(st.get("max_step_time", 0.0))
            self.routing_analysis_failures = int(st.get("routing_analysis_failures", 0))
            if "rng" in st:
                try:
                    _set_rng_state(st["rng"])
                except Exception:
                    logger.exception("RNG state not restored (training continues)")

        hist = checkpoint_dir / "routing_history.json"
        if hist.exists():
            with open(hist) as f:
                self.routing_history = [RoutingSnapshot(**h) for h in json.load(f)]

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"  State: epoch={self.state.epoch}, samples_seen={self.state.samples_seen}, "
                    f"global_step={self.state.global_step}")

    # ------------------------------------------------------------- HF pushes

    def _maybe_push_best(self, force: bool = False, blocking: bool = False) -> None:
        if not self._best_dirty:
            return
        step = self.state.global_step
        if not force and self._last_best_push_step is not None and \
                step - self._last_best_push_step < 4 * self.save_steps:
            return
        if self.pusher.submit_dir("best_model", blocking=blocking):
            self._best_dirty = False
            self._last_best_push_step = step

    def _push_on_deadline(self) -> None:
        """Blocking: latest/ (resume state) + best_model/ if changed since last push. Retries are
        bounded by the time SLURM leaves after GLR_DEADLINE (sbatch: deadline = end - 600 s)."""
        self.pusher.budget_s = 240.0
        try:
            self._maybe_push_best(force=True, blocking=True)
            self.pusher.submit_dir("latest", blocking=True)
            self.pusher.wait()
        finally:
            self.pusher.budget_s = None

    def finalize_push(self) -> None:
        """End of run (blocking): ONE commit with best_model, final_model, latest,
        visualizations, the root result files and TRAINING_DONE, retried (rate limits) until
        shortly before the deadline. Called by the pipeline after final_results.json is
        written. On failure the outputs stay on the node (train.sbatch keeps them when
        TRAINING_DONE is not on the Hub)."""
        try:
            if self.pusher.enabled:
                budget = (self.deadline - time.time() - 60.0) if self.deadline else 3600.0
                files = [self.output_dir / f for f in (
                    "final_results.json", "eval_results.json", "routing_history.json",
                    "final_examples.npz", "generation_results.json", "experiment_config.json",
                    "data_stats.json", "frozen_gate_calibration.json", TRAINING_DONE)]
                ok = self.pusher.commit_final(
                    ["best_model", "final_model", "latest", "visualizations"], files,
                    budget_s=max(budget, 120.0))
                if not ok:
                    logger.error("TRAINING_DONE could not be pushed/verified on the Hub: "
                                 "the chain may resubmit this run (outputs kept on the node)")
        finally:
            self.pusher.shutdown()

    def shutdown_push(self) -> None:
        self.pusher.shutdown()

    def push_latest_after_crash(self) -> None:
        """Blocking push of the local latest/ (last periodic save, consistent by construction).
        Never raises: the caller re-raises the original error."""
        try:
            if self.pusher.enabled and (self.output_dir / "latest").is_dir():
                logger.warning("Training crashed: pushing the last local latest/ before exiting")
                self.pusher.submit_dir("latest", blocking=True)
            self.pusher.shutdown()
        except Exception:
            logger.exception("latest/ push after crash failed")

    def _install_sigterm_handler(self) -> None:
        """SIGTERM (scancel, forwarded by train.sbatch) -> stop at the next optimizer step through
        the deadline path (save + blocking push of latest/)."""
        def _handler(signum, frame):
            logger.warning("SIGTERM received: stopping after the current optimizer step")
            self._term_requested = True
        try:
            self._prev_sigterm = signal.signal(signal.SIGTERM, _handler)
        except ValueError:  # not the main thread
            self._prev_sigterm = None

    def _restore_sigterm_handler(self) -> None:
        """After the loop: final eval / pushes are not interruptible at a step boundary, so TERM
        goes back to the default (process ends; latest/ was saved at the last save_steps)."""
        prev = getattr(self, "_prev_sigterm", None)
        if prev is not None:
            try:
                signal.signal(signal.SIGTERM, prev)
            except ValueError:
                pass
            self._prev_sigterm = None

    # ------------------------------------------------------------------ main

    def _stop_for_deadline(self) -> bool:
        if self.stop_at_step and self.state.global_step >= self.stop_at_step:
            logger.warning(f"GLR_STOP_AT_STEP={self.stop_at_step} reached (test hook)")
            self.stop_at_step = 0  # once per process
            return True
        return self._term_requested or not self.time_allows(0.0)

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Main loop: optimizer steps until `max_steps` or the deadline.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from, or "auto" to find latest
        """
        logger.info("=" * 60)
        logger.info("Starting Gated LoRA Training" if self.is_gated else "Starting LoRA baseline training")
        logger.info("=" * 60)
        self._init_deadline()
        self._install_sigterm_handler()

        if self.is_training_done():
            logger.info(f"{TRAINING_DONE} present in {self.output_dir}: nothing to do")
            return {"status": "already_complete", "total_steps": self.state.global_step}

        if resume_from_checkpoint:
            checkpoint_path = None
            if resume_from_checkpoint == "auto":
                checkpoint_path = self.find_latest_checkpoint()
                if checkpoint_path:
                    logger.info(f"Auto-detected checkpoint: {checkpoint_path}")
                else:
                    logger.info("No checkpoint found, starting from scratch")
            else:
                checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path and checkpoint_path.exists():
                self.load_checkpoint(str(checkpoint_path))
                logger.info(f"Resuming at step {self.state.global_step}, epoch {self.state.epoch}, "
                            f"sample {self.state.samples_seen}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info(f"Initial VRAM - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                        f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

        start_time = time.time()
        self.model.train()
        self._set_model_attr("collect_routing_stats", self.collect_routing_stats)
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar = None
        if TQDM_AVAILABLE:
            progress_bar = tqdm(total=self.max_steps, initial=self.state.global_step,
                                desc="train", unit="step", dynamic_ncols=True, mininterval=30)

        window = 0  # micro-batches in the current accumulation window (carried across epochs)
        loss_sum: Optional[torch.Tensor] = None  # detached [4] sums since last log
        n_micro = 0
        grad_norm: Optional[torch.Tensor] = None
        window_t0 = time.time()
        hit_deadline = False
        steps_this_job = 0

        if self.state.global_step < self.max_steps and self._stop_for_deadline():
            hit_deadline = True

        while self.state.global_step < self.max_steps and not hit_deadline:
            loader = self._train_loader()
            got_batch = False
            for batch in loader:
                got_batch = True
                if window == 0:
                    window_t0 = time.time()
                    self._update_reg_scale()
                stats = self.train_step(batch)
                am = batch["attention_mask"]  # CPU: no sync
                self._tok_real += int(am.sum())
                self._tok_total += am.numel()
                loss_sum = stats if loss_sum is None else loss_sum + stats
                n_micro += 1
                bsz = int(batch["input_ids"].size(0))
                self.state.samples_seen += bsz
                self.state.total_samples_seen += bsz
                self.state.batch_idx += 1
                window += 1
                if window < self.gradient_accumulation_steps:
                    continue

                # ---- optimizer step (full window) ----
                window = 0
                grad_norm = self._optimizer_step()
                self.state.global_step += 1
                steps_this_job += 1
                step_time = time.time() - window_t0
                if self._first_step_time is None:
                    self._first_step_time = step_time
                    if torch.cuda.is_available():
                        logger.info(f"Peak VRAM after first step: "
                                    f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
                else:
                    self.max_step_time = max(self.max_step_time, step_time)
                if progress_bar is not None:
                    progress_bar.update(1)
                step = self.state.global_step

                if step % self.logging_steps == 0:
                    self._log_train(loss_sum, n_micro, grad_norm, progress_bar)
                    loss_sum, n_micro = None, 0

                if step % self.eval_steps == 0 and step < self.max_steps:
                    self._maybe_eval_for_selection()

                if step % self.save_steps == 0 and step < self.max_steps:
                    self.save_checkpoint("latest")

                if (self.is_gated and self.routing_analysis_steps > 0
                        and step % self.routing_analysis_steps == 0):
                    self._routing_analysis_safely()

                self._maybe_push_best()

                if step >= self.max_steps:
                    break
                if self._stop_for_deadline():
                    hit_deadline = True
                    break
            else:
                # Epoch exhausted: leftovers in `window` carry into the next epoch.
                if not got_batch and self.state.samples_seen == 0:
                    raise RuntimeError(f"Train loader yielded no batch for epoch {self.state.epoch}")
                logger.info(f"Epoch {self.state.epoch} done ({self.state.samples_seen} samples)")
                self.state.epoch += 1
                self.state.samples_seen = 0
                self.state.batch_idx = 0

        if progress_bar is not None:
            progress_bar.close()
        if loss_sum is not None:
            self._log_train(loss_sum, n_micro, grad_norm, None)
        if not hit_deadline:
            self._restore_sigterm_handler()

        if hit_deadline:
            return self._exit_for_deadline(start_time, "training")

        # ---- end of budget: selection eval, full evals, final save ----
        self._maybe_eval_for_selection()
        full_loaders = [self._eval_loader("val", None), self._eval_loader("final", None)]
        need = sum(self._eval_seconds_estimate(l) for l in full_loaders)
        if not self.time_allows(need):
            logger.warning(f"Final evaluation (~{need / 60:.1f} min) does not fit before the deadline")
            return self._exit_for_deadline(start_time, "final evaluation")

        final_val = self.evaluate("val", None)
        self._log_eval(final_val, "final_val")
        final_test = self.evaluate("final", None, dump_path=self.output_dir / "final_examples.npz")
        self._log_eval(final_test, "final_test")

        generation: Dict[str, Any] = {}
        if self.generation_tasks:
            if self.time_allows(self._generation_seconds_estimate()):
                generation = self.generation_eval("final")
                with open(self.output_dir / "generation_results.json", "w") as f:
                    json.dump(generation, f)
            else:
                logger.warning("Generation eval does not fit before the deadline")
                return self._exit_for_deadline(start_time, "generation evaluation")

        self._save_weights_dir("final_model")
        self.save_checkpoint("latest")
        eval_results = {
            "val_full": final_val,
            "final": final_test,
            "generation": {t: {k: v for k, v in r.items()
                               if k not in ("per_example", "predictions", "example_idx")}
                           for t, r in generation.items()},
            "best_eval_loss": self.state.best_eval_loss,
            "best_eval_step": self.state.best_eval_step,
            "selection_metric": "val mean task answer loss "
                                f"({self.eval_samples_per_task} samples/task)",
        }
        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        self._save_routing_history()
        self._mark_training_done()

        elapsed_time = time.time() - start_time
        logger.info(f"\nTraining completed ({steps_this_job} steps this job, {elapsed_time / 60:.2f} min)")
        return {
            "status": "completed",
            "final_train_loss": self.state.total_train_loss / max(self.state.num_train_steps, 1),
            "final_eval_metrics": eval_results,
            "best_eval_loss": self.state.best_eval_loss,
            "best_eval_step": self.state.best_eval_step,
            "total_steps": self.state.global_step,
            "total_samples_seen": self.state.total_samples_seen,
            "training_time_minutes": elapsed_time / 60,
            "max_step_seconds": self.max_step_time,
            "num_routing_snapshots": len(self.routing_history),
            "routing_analysis_failures": self.routing_analysis_failures,
            "precision": self.precision,
        }

    def _exit_for_deadline(self, start_time: float, phase: str) -> Dict[str, Any]:
        elapsed = time.time() - start_time
        logger.info(f"\n{'=' * 60}")
        logger.info(f"DEADLINE: stopping during {phase} at step {self.state.global_step} "
                    f"({elapsed / 60:.1f} min this job, max step {self.max_step_time:.2f}s)")
        logger.info(f"{'=' * 60}")
        self.save_checkpoint("latest")
        self._push_on_deadline()
        logger.info("Resume with --resume auto to continue training")
        return {
            "status": "time_limit",
            "phase": phase,
            "current_epoch": self.state.epoch,
            "samples_seen": self.state.samples_seen,
            "global_step": self.state.global_step,
            "training_time_minutes": elapsed / 60,
        }

    def _mark_training_done(self):
        """Create a marker file indicating training is complete."""
        done_file = self.output_dir / TRAINING_DONE
        with open(done_file, "w") as f:
            f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total steps: {self.state.global_step}\n")
            f.write(f"Best eval loss: {self.state.best_eval_loss} (step {self.state.best_eval_step})\n")
        logger.info(f"Created {done_file}")

    def is_training_done(self) -> bool:
        """Check if training is already complete."""
        return (self.output_dir / TRAINING_DONE).exists()


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Any,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create optimizer and learning rate scheduler."""

    # Separate parameters for different learning rates
    gating_params = []
    expert_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "gating_network" in name:  # NOT "gate": that also matches the gate_proj experts
            gating_params.append(param)
        elif "expert" in name.lower() or "lora" in name.lower():
            expert_params.append(param)
        else:
            other_params.append(param)

    # Parameter groups with potentially different LRs
    lr = config.training.learning_rate
    param_groups = []

    if gating_params:
        param_groups.append({"params": gating_params, "lr": lr, "name": "gating"})
    if expert_params:
        param_groups.append({"params": expert_params, "lr": lr, "name": "experts"})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr, "name": "other"})

    # Create optimizer
    optimizer_name = getattr(config.training, "optimizer", "adamw").lower()
    weight_decay = getattr(config.training, "weight_decay", 0.01)

    # fused AdamW on CUDA: one kernel, and GradScaler passes found_inf to it instead of a
    # host sync per step (all trainable params are fp32 CUDA tensors, checked by the trainer)
    on_cuda = all(p.is_cuda for g in param_groups for p in g["params"])
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay,
                                      **({"fused": True} if on_cuda else {"foreach": True}))
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # Create scheduler
    scheduler_name = getattr(config.training, "scheduler", "cosine").lower()
    warmup_ratio = getattr(config.training, "warmup_ratio", 0.1)
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    if scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_training_steps - warmup_steps),
            eta_min=lr * 0.1,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    elif scheduler_name == "linear":
        from torch.optim.lr_scheduler import LinearLR

        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps,
        )
    else:
        scheduler = None

    logger.info(f"Created optimizer: {optimizer_name}, scheduler: {scheduler_name}")
    logger.info(f"Parameter groups: gating={len(gating_params)}, experts={len(expert_params)}, other={len(other_params)}")

    return optimizer, scheduler


if __name__ == "__main__":
    # Quick test
    print("GatedLoRATrainer module loaded successfully")
    print(f"WandB available: {WANDB_AVAILABLE}")
    print(f"Default max runtime: {DEFAULT_MAX_RUNTIME_SECONDS / 3600:.2f} hours")
