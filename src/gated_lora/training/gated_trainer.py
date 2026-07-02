"""
GatedLoRATrainer - Unified trainer combining 2-phase training, SLURM chaining,
and periodic routing analysis.

Features:
1. Gating Warmup Phase: Freeze experts, train only gating network
2. Joint Training Phase: Train both gating and experts
3. Routing Statistics Logging: Expert usage, entropy, load imbalance
4. Load Balancing Loss: Auxiliary loss for expert diversity
5. Timer-based checkpointing for SLURM job chaining (4h partition limit)
6. Resume from checkpoint with batch skipping
7. Periodic Routing Analysis: per-layer, per-task routing snapshots during training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
import json
import math
import os
import re

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

# Default max runtime: 3h30 (leaving 30min margin for 4h SLURM limit).
# Override per-job with env GLR_MAX_RUNTIME_SECONDS (set by train.sbatch)
# so partitions with different MaxTime don't need a code change.
DEFAULT_MAX_RUNTIME_SECONDS = float(
    os.environ.get("GLR_MAX_RUNTIME_SECONDS", 3.5 * 3600)
)


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
    """Tracks training progress."""
    global_step: int = 0
    epoch: int = 0
    batch_idx: int = 0  # NEW: batch index within current epoch for resume
    best_eval_loss: float = float("inf")
    warmup_completed: bool = False
    total_train_loss: float = 0.0
    total_lb_loss: float = 0.0  # Load balancing loss
    num_train_steps: int = 0


@dataclass
class RoutingSnapshot:
    """Snapshot of routing patterns at a given step."""
    step: int
    epoch: int
    layer_expert_usage: List[List[float]]  # [num_layers, num_experts]
    task_layer_expert_usage: Dict[str, List[List[float]]]  # {task: [num_layers, num_experts]}
    layer_entropy: List[float]  # [num_layers]
    specialization_scores: Dict[str, float]  # per-layer specialization scores


class GatedLoRATrainer:
    """
    Custom trainer for Gated LoRA with:
    - 2-phase training (gating warmup → joint)
    - Routing statistics logging
    - Load balancing loss
    - Timer-based checkpointing for SLURM job chaining
    - Resume from checkpoint with batch skipping
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
        output_dir: str = "./outputs",
        device: str = "cuda",
        max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,  # NEW
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_runtime_seconds = max_runtime_seconds  # NEW

        # Training state
        self.state = TrainingState()

        # Extract config values
        self.num_epochs = getattr(config.training, "num_epochs", 4) if config else 4
        self.gating_warmup_steps = getattr(config.training, "gating_warmup_steps", 0) if config else 0
        self.gating_warmup_epochs = getattr(config.training, "gating_warmup_epochs", 0) if config else 0
        self.freeze_experts_during_warmup = getattr(config.training, "freeze_experts_during_warmup", True) if config else True
        self.use_load_balancing = getattr(config.model, "use_load_balancing", False) if config else False
        self.load_balancing_weight = getattr(config.model, "load_balancing_weight", 0.001) if config else 0.001
        self.log_routing_stats = getattr(config.training, "log_routing_stats", True) if config else True
        self.logging_steps = getattr(config.training, "logging_steps", 10) if config else 10
        self.eval_steps = getattr(config.training, "eval_steps", 500) if config else 500
        self.save_steps = getattr(config.training, "save_steps", 500) if config else 500
        self.gradient_accumulation_steps = getattr(config.training, "gradient_accumulation_steps", 1) if config else 1
        self.max_grad_norm = getattr(config.training, "max_grad_norm", 1.0) if config else 1.0

        # Routing statistics accumulator
        self.routing_stats_buffer: List[Dict[str, float]] = []

        # Check if model is gated
        self.is_gated = hasattr(model, "gating_network") or hasattr(model, "is_gated")

        # NEW: Periodic routing analysis for convergence study
        self.routing_analysis_steps = getattr(config.training, "routing_analysis_steps", 500) if config else 500
        self.routing_history: List[RoutingSnapshot] = []
        self.analysis_dataloader = None  # Will be set if multi-task data available

        logger.info(f"GatedLoRATrainer initialized:")
        if self.gating_warmup_epochs > 0:
            logger.info(f"  - Gating warmup: {self.gating_warmup_epochs} epoch(s)")
        else:
            logger.info(f"  - Gating warmup: {self.gating_warmup_steps} steps")
        logger.info(f"  - Freeze experts during warmup: {self.freeze_experts_during_warmup}")
        logger.info(f"  - Load balancing: {self.use_load_balancing} (weight={self.load_balancing_weight})")
        logger.info(f"  - Is gated model: {self.is_gated}")
        logger.info(f"  - Max runtime: {self.max_runtime_seconds / 3600:.2f} hours")
        logger.info(f"  - Routing analysis every: {self.routing_analysis_steps} steps")

    def set_analysis_dataloader(self, dataloader: DataLoader):
        """Set the dataloader used for periodic routing analysis."""
        self.analysis_dataloader = dataloader
        logger.info(f"Analysis dataloader set with {len(dataloader)} batches")

    def _freeze_experts(self):
        """Freeze LoRA expert parameters during gating warmup."""
        if not self.is_gated:
            return

        frozen_count = 0
        if hasattr(self.model, "expert_pools"):
            for pool in self.model.expert_pools:
                for param in pool.parameters():
                    param.requires_grad = False
                    frozen_count += 1
        elif hasattr(self.model, "lora_layers"):
            for param in self.model.lora_layers.parameters():
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Froze {frozen_count} expert parameters for gating warmup")

    def _unfreeze_experts(self):
        """Unfreeze LoRA expert parameters after warmup."""
        if not self.is_gated:
            return

        unfrozen_count = 0
        if hasattr(self.model, "expert_pools"):
            for pool in self.model.expert_pools:
                for param in pool.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
        elif hasattr(self.model, "lora_layers"):
            for param in self.model.lora_layers.parameters():
                param.requires_grad = True
                unfrozen_count += 1

        logger.info(f"Unfroze {unfrozen_count} expert parameters - starting joint training")

    def _extract_routing_stats(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """Extract routing statistics from model outputs."""
        stats = {}

        # Check for routing info in outputs
        if "routing_info" in outputs:
            routing_info = outputs["routing_info"]

            # Expert usage
            if "expert_usage" in routing_info:
                usage = routing_info["expert_usage"]
                if isinstance(usage, torch.Tensor):
                    usage = usage.detach().cpu()
                for i, u in enumerate(usage):
                    stats[f"expert_{i}_usage"] = float(u)

            # Entropy
            if "entropy" in routing_info:
                entropy = routing_info["entropy"]
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.detach().cpu().item()
                stats["routing_entropy"] = float(entropy)

            # Normalized entropy
            if "normalized_entropy" in routing_info:
                norm_ent = routing_info["normalized_entropy"]
                if isinstance(norm_ent, torch.Tensor):
                    norm_ent = norm_ent.detach().cpu().item()
                stats["routing_normalized_entropy"] = float(norm_ent)

            # Load imbalance
            if "load_imbalance" in routing_info:
                imb = routing_info["load_imbalance"]
                if isinstance(imb, torch.Tensor):
                    imb = imb.detach().cpu().item()
                stats["load_imbalance"] = float(imb)

            # Top-1 dominance
            if "top1_dominance" in routing_info:
                dom = routing_info["top1_dominance"]
                if isinstance(dom, torch.Tensor):
                    dom = dom.detach().cpu().item()
                stats["top1_dominance"] = float(dom)

        # Load balancing loss
        if "load_balancing_loss" in outputs:
            lb_loss = outputs["load_balancing_loss"]
            if isinstance(lb_loss, torch.Tensor):
                lb_loss = lb_loss.detach().cpu().item()
            stats["load_balancing_loss"] = float(lb_loss)

        return stats

    def _aggregate_routing_stats(self) -> Dict[str, float]:
        """Aggregate buffered routing statistics."""
        if not self.routing_stats_buffer:
            return {}

        aggregated = {}
        keys = self.routing_stats_buffer[0].keys()

        for key in keys:
            values = [s[key] for s in self.routing_stats_buffer if key in s]
            if values:
                aggregated[f"avg_{key}"] = sum(values) / len(values)

        self.routing_stats_buffer.clear()
        return aggregated

    def run_routing_analysis(self, num_batches: int = 20) -> Optional[RoutingSnapshot]:
        """
        Run detailed routing analysis and create a snapshot.

        Args:
            num_batches: Number of batches to analyze

        Returns:
            RoutingSnapshot with per-layer, per-task routing patterns
        """
        if not self.is_gated or self.analysis_dataloader is None:
            return None

        self.model.eval()

        # Get model dimensions
        num_experts = getattr(self.model, "num_experts", 3)
        num_layers = getattr(self.model, "num_layers", 32)

        # Accumulators: layer_expert_counts[layer][expert] = count
        layer_expert_counts = [[0.0] * num_experts for _ in range(num_layers)]
        task_layer_expert_counts: Dict[str, List[List[float]]] = {}
        total_samples = 0
        task_samples: Dict[str, int] = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.analysis_dataloader):
                if batch_idx >= num_batches:
                    break

                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                batch_tasks = batch.get("task", None)
                batch_size = input_ids.size(0)
                total_samples += batch_size

                # Forward with routing info
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_routing_info=True,
                )

                if isinstance(outputs, dict) and "routing_info" in outputs:
                    routing_info = outputs["routing_info"]

                    if "per_layer_info" in routing_info:
                        per_layer_info = routing_info["per_layer_info"]

                        for layer_idx_key, layer_info in per_layer_info.items():
                            layer_idx = int(layer_idx_key) if isinstance(layer_idx_key, str) else layer_idx_key

                            if "gate_weights" in layer_info:
                                gate_weights = layer_info["gate_weights"]  # [batch, num_experts]

                                # Aggregate per-layer
                                mean_weights = gate_weights.mean(dim=0).cpu().tolist()
                                for e, w in enumerate(mean_weights):
                                    layer_expert_counts[layer_idx][e] += w * batch_size

                                # Aggregate per-task
                                if batch_tasks is not None:
                                    for sample_idx, task in enumerate(batch_tasks):
                                        if task not in task_layer_expert_counts:
                                            task_layer_expert_counts[task] = [[0.0] * num_experts for _ in range(num_layers)]
                                            task_samples[task] = 0

                                        sample_weights = gate_weights[sample_idx].cpu().tolist()
                                        for e, w in enumerate(sample_weights):
                                            task_layer_expert_counts[task][layer_idx][e] += w

                                        if layer_idx == 0:  # Count only once per sample
                                            task_samples[task] = task_samples.get(task, 0) + 1

        # Normalize
        if total_samples > 0:
            for layer_idx in range(num_layers):
                total = sum(layer_expert_counts[layer_idx])
                if total > 0:
                    layer_expert_counts[layer_idx] = [c / total for c in layer_expert_counts[layer_idx]]

        # Normalize per-task
        task_layer_expert_usage: Dict[str, List[List[float]]] = {}
        for task, counts in task_layer_expert_counts.items():
            task_layer_expert_usage[task] = []
            for layer_idx in range(num_layers):
                total = sum(counts[layer_idx])
                if total > 0:
                    task_layer_expert_usage[task].append([c / total for c in counts[layer_idx]])
                else:
                    task_layer_expert_usage[task].append([1.0 / num_experts] * num_experts)

        # Compute entropy per layer
        layer_entropy = []
        for layer_idx in range(num_layers):
            probs = layer_expert_counts[layer_idx]
            entropy = 0.0
            for p in probs:
                if p > 0:
                    entropy -= p * math.log(p + 1e-10)
            layer_entropy.append(entropy)

        # Compute specialization score per layer (variance across tasks)
        specialization_scores = {}
        for layer_idx in range(num_layers):
            if len(task_layer_expert_usage) >= 2:
                # For each expert, compute variance across tasks
                expert_variances = []
                for e in range(num_experts):
                    task_usages = [task_layer_expert_usage[t][layer_idx][e] for t in task_layer_expert_usage]
                    if len(task_usages) >= 2:
                        mean_usage = sum(task_usages) / len(task_usages)
                        variance = sum((u - mean_usage) ** 2 for u in task_usages) / len(task_usages)
                        expert_variances.append(variance)

                if expert_variances:
                    specialization_scores[f"layer_{layer_idx}"] = sum(expert_variances) / len(expert_variances)
                else:
                    specialization_scores[f"layer_{layer_idx}"] = 0.0
            else:
                specialization_scores[f"layer_{layer_idx}"] = 0.0

        self.model.train()

        snapshot = RoutingSnapshot(
            step=self.state.global_step,
            epoch=self.state.epoch,
            layer_expert_usage=layer_expert_counts,
            task_layer_expert_usage=task_layer_expert_usage,
            layer_entropy=layer_entropy,
            specialization_scores=specialization_scores,
        )

        return snapshot

    def _save_routing_history(self):
        """Save routing history to file."""
        if not self.routing_history:
            return

        history_path = self.output_dir / "routing_history.json"

        # Convert to serializable format
        history_data = []
        for snapshot in self.routing_history:
            history_data.append({
                "step": snapshot.step,
                "epoch": snapshot.epoch,
                "layer_expert_usage": snapshot.layer_expert_usage,
                "task_layer_expert_usage": snapshot.task_layer_expert_usage,
                "layer_entropy": snapshot.layer_entropy,
                "specialization_scores": snapshot.specialization_scores,
            })

        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Saved {len(history_data)} routing snapshots to {history_path}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Execute a single training step.

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels", batch["input_ids"]),
        )

        # Get loss
        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("lm_loss"))
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs[0]

        # Add load balancing loss if enabled and in joint training phase
        metrics = {}
        if self.use_load_balancing and self.state.warmup_completed:
            if isinstance(outputs, dict) and "load_balancing_loss" in outputs:
                lb_loss = outputs["load_balancing_loss"]
                loss = loss + self.load_balancing_weight * lb_loss
                metrics["load_balancing_loss"] = lb_loss.item() if isinstance(lb_loss, torch.Tensor) else lb_loss

        # Backward pass
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Extract routing stats
        if self.log_routing_stats and self.is_gated:
            if isinstance(outputs, dict):
                routing_stats = self._extract_routing_stats(outputs)
                if routing_stats:
                    self.routing_stats_buffer.append(routing_stats)

        metrics["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss
        return loss.item(), metrics

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Execute a single evaluation step."""
        self.model.eval()

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch["input_ids"]),
            )

        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("lm_loss"))
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs[0]

        metrics = {"eval_loss": loss.item() if isinstance(loss, torch.Tensor) else loss}

        # Extract routing stats for eval
        if self.log_routing_stats and self.is_gated and isinstance(outputs, dict):
            routing_stats = self._extract_routing_stats(outputs)
            for k, v in routing_stats.items():
                metrics[f"eval_{k}"] = v

        return loss.item(), metrics

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataloader."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = []

        for batch in self.eval_dataloader:
            loss, metrics = self.eval_step(batch)
            total_loss += loss
            num_batches += 1
            all_metrics.append(metrics)

        avg_loss = total_loss / max(num_batches, 1)

        # Aggregate metrics
        result = {"eval_loss": avg_loss}
        if all_metrics:
            for key in all_metrics[0].keys():
                if key != "eval_loss":
                    values = [m[key] for m in all_metrics if key in m]
                    if values:
                        result[key] = sum(values) / len(values)

        # Compute perplexity
        result["eval_perplexity"] = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

        return result

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in output_dir."""
        if not self.output_dir.exists():
            return None

        # Look for checkpoint-* directories
        checkpoint_dirs = []
        for d in self.output_dir.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                # Extract step number
                match = re.match(r"checkpoint-(\d+)", d.name)
                if match:
                    step = int(match.group(1))
                    checkpoint_dirs.append((step, d))

        # Also check for "latest" checkpoint (saved on timeout)
        latest_dir = self.output_dir / "latest"
        if latest_dir.exists() and (latest_dir / "training_state.json").exists():
            # Read step from training_state.json
            with open(latest_dir / "training_state.json", "r") as f:
                state = json.load(f)
                step = state.get("global_step", 0)
                checkpoint_dirs.append((step, latest_dir))

        if not checkpoint_dirs:
            return None

        # Return the one with highest step
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_dirs[0][1]

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop with 2-phase training.

        Phase 1 (Warmup): Train only gating network
        Phase 2 (Joint): Train both gating and experts

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from, or "auto" to find latest
        """
        logger.info("=" * 60)
        logger.info("Starting Gated LoRA Training")
        logger.info("=" * 60)

        # Handle resume
        start_epoch = 0
        start_batch_idx = 0

        if resume_from_checkpoint:
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
                start_epoch = self.state.epoch
                start_batch_idx = self.state.batch_idx
                logger.info(f"Resuming from epoch {start_epoch + 1}, batch {start_batch_idx}")
                logger.info(f"  Global step: {self.state.global_step}")
                logger.info(f"  Warmup completed: {self.state.warmup_completed}")

        # Check if training is already complete
        if start_epoch >= self.num_epochs:
            logger.info("Training already complete!")
            self._mark_training_done()
            return {
                "status": "already_complete",
                "final_train_loss": self.state.total_train_loss / max(self.state.num_train_steps, 1),
                "best_eval_loss": self.state.best_eval_loss,
                "total_steps": self.state.global_step,
            }

        # Phase 1: Gating Warmup (if enabled and not already completed)
        warmup_enabled = (self.gating_warmup_epochs > 0 or self.gating_warmup_steps > 0) and self.freeze_experts_during_warmup and self.is_gated
        if warmup_enabled and not self.state.warmup_completed:
            if self.gating_warmup_epochs > 0:
                logger.info(f"\n--- Phase 1: Gating Warmup ({self.gating_warmup_epochs} epoch(s)) ---")
            else:
                logger.info(f"\n--- Phase 1: Gating Warmup ({self.gating_warmup_steps} steps) ---")
            self._freeze_experts()

        total_batches_per_epoch = len(self.train_dataloader)
        total_steps = total_batches_per_epoch * self.num_epochs
        logger.info(f"Total batches per epoch: {total_batches_per_epoch}")
        logger.info(f"Total training batches: {total_steps}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

        # Log initial VRAM usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"Initial VRAM - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        start_time = time.time()
        self.optimizer.zero_grad()

        # Track if we hit the time limit
        hit_time_limit = False

        for epoch in range(start_epoch, self.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")

            # Determine starting batch for this epoch
            skip_batches = start_batch_idx if epoch == start_epoch else 0
            if skip_batches > 0:
                logger.info(f"Skipping first {skip_batches} batches (already processed)")

            # Create progress bar for this epoch
            if TQDM_AVAILABLE:
                progress_bar = tqdm(
                    enumerate(self.train_dataloader),
                    total=total_batches_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=True,
                    initial=skip_batches,  # Start progress bar at correct position
                )
            else:
                progress_bar = enumerate(self.train_dataloader)

            # Check for epoch-based warmup completion at start of epoch
            if (not self.state.warmup_completed and
                self.gating_warmup_epochs > 0 and
                epoch >= self.gating_warmup_epochs and
                self.is_gated):

                logger.info(f"\n--- Phase 2: Joint Training (after {self.gating_warmup_epochs} warmup epoch(s)) ---")
                self._unfreeze_experts()
                self.state.warmup_completed = True

            for batch_idx, batch in progress_bar:
                # Skip batches if resuming mid-epoch
                if batch_idx < skip_batches:
                    continue

                # Update batch_idx in state for checkpointing
                self.state.batch_idx = batch_idx

                # Check for step-based warmup completion (only if not using epoch-based)
                if (not self.state.warmup_completed and
                    self.gating_warmup_epochs == 0 and
                    self.state.global_step >= self.gating_warmup_steps and
                    self.gating_warmup_steps > 0 and
                    self.is_gated):

                    logger.info(f"\n--- Phase 2: Joint Training (step {self.state.global_step}) ---")
                    self._unfreeze_experts()
                    self.state.warmup_completed = True

                # Training step
                loss, metrics = self.train_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                self.state.total_train_loss += loss
                self.state.num_train_steps += 1

                # Log VRAM after first batch
                if batch_idx == skip_batches and epoch == start_epoch and torch.cuda.is_available():
                    peak_vram = torch.cuda.max_memory_allocated() / 1e9
                    logger.info(f"Peak VRAM after first batch: {peak_vram:.2f}GB / 47GB")

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.state.global_step += 1

                    # Logging
                    if self.state.global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.optimizer.param_groups[0]["lr"]
                        phase = "warmup" if not self.state.warmup_completed else "joint"
                        elapsed = time.time() - start_time
                        remaining = self.max_runtime_seconds - elapsed

                        log_dict = {
                            "step": self.state.global_step,
                            "epoch": epoch + 1,
                            "batch_idx": batch_idx,
                            "loss": avg_loss,
                            "learning_rate": lr,
                            "phase": phase,
                            "elapsed_minutes": elapsed / 60,
                            "remaining_minutes": remaining / 60,
                        }

                        # Add routing stats
                        if self.routing_stats_buffer:
                            routing_stats = self._aggregate_routing_stats()
                            log_dict.update(routing_stats)

                        # Update tqdm progress bar with metrics
                        if TQDM_AVAILABLE and hasattr(progress_bar, 'set_postfix'):
                            progress_bar.set_postfix({
                                'loss': f'{avg_loss:.4f}',
                                'lr': f'{lr:.2e}',
                                'remain': f'{remaining/60:.0f}m',
                            })

                        logger.info(
                            f"Step {self.state.global_step} (batch {batch_idx}): loss={avg_loss:.4f}, "
                            f"lr={lr:.2e}, phase={phase}, remaining={remaining/60:.1f}min"
                        )

                        # WandB logging
                        if WANDB_AVAILABLE and wandb.run is not None:
                            wandb.log(log_dict, step=self.state.global_step)

                    # Evaluation — skipped when close to the runtime budget:
                    # a full eval can take 15-30 min, and overshooting the
                    # SLURM wall means SIGKILL with no "latest" checkpoint.
                    eval_time_ok = (
                        time.time() - start_time
                    ) < self.max_runtime_seconds - 20 * 60
                    if (self.state.global_step % self.eval_steps == 0
                            and self.eval_dataloader is not None
                            and eval_time_ok):
                        eval_metrics = self.evaluate()
                        logger.info(f"Eval at step {self.state.global_step}: {eval_metrics}")

                        if WANDB_AVAILABLE and wandb.run is not None:
                            wandb.log(eval_metrics, step=self.state.global_step)

                        # Track best model
                        if eval_metrics["eval_loss"] < self.state.best_eval_loss:
                            self.state.best_eval_loss = eval_metrics["eval_loss"]
                            self.save_checkpoint("best_model")

                    # Checkpoint saving
                    if self.state.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.state.global_step}")

                    # Periodic routing analysis for convergence study
                    if (self.state.global_step % self.routing_analysis_steps == 0 and
                        self.is_gated and self.analysis_dataloader is not None):
                        logger.info(f"Running routing analysis at step {self.state.global_step}...")
                        snapshot = self.run_routing_analysis(num_batches=20)
                        if snapshot is not None:
                            self.routing_history.append(snapshot)
                            max_spec_layer = max(snapshot.specialization_scores.items(), key=lambda x: x[1])
                            logger.info(f"  Max specialization: {max_spec_layer[0]} = {max_spec_layer[1]:.4f}")
                            if WANDB_AVAILABLE and wandb.run is not None:
                                wandb.log({
                                    "routing/max_specialization_score": max_spec_layer[1],
                                    "routing/max_specialization_layer": int(max_spec_layer[0].split("_")[1]),
                                    "routing/mean_entropy": sum(snapshot.layer_entropy) / len(snapshot.layer_entropy),
                                }, step=self.state.global_step)

                # Check time limit - save and exit if approaching limit
                elapsed = time.time() - start_time
                if elapsed >= self.max_runtime_seconds:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"TIME LIMIT REACHED ({elapsed/3600:.2f} hours)")
                    logger.info(f"Saving checkpoint and exiting...")
                    logger.info(f"{'='*60}")

                    # Save checkpoint with next batch position
                    self.state.batch_idx = batch_idx + 1
                    if self.state.batch_idx >= total_batches_per_epoch:
                        # Move to next epoch
                        self.state.epoch = epoch + 1
                        self.state.batch_idx = 0

                    self.save_checkpoint("latest")
                    hit_time_limit = True
                    break

            if hit_time_limit:
                break

            # End of epoch - reset batch_idx for next epoch
            self.state.batch_idx = 0

            # End of epoch evaluation (same runtime guard as step-based eval)
            eval_time_ok = (time.time() - start_time) < self.max_runtime_seconds - 20 * 60
            eval_metrics = self.evaluate() if (self.eval_dataloader and eval_time_ok) else {}
            epoch_avg_loss = epoch_loss / max(epoch_steps, 1)

            logger.info(f"Epoch {epoch + 1} completed: avg_loss={epoch_avg_loss:.4f}")
            if eval_metrics:
                logger.info(f"Eval metrics: {eval_metrics}")

            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch + 1}")

        # Training complete or time limit hit
        elapsed_time = time.time() - start_time

        if hit_time_limit:
            logger.info(f"\nJob ended due to time limit after {elapsed_time / 60:.2f} minutes")
            logger.info(f"Resume with --resume auto to continue training")
            return {
                "status": "time_limit",
                "current_epoch": self.state.epoch,
                "current_batch": self.state.batch_idx,
                "global_step": self.state.global_step,
                "training_time_minutes": elapsed_time / 60,
            }

        logger.info(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")

        # Final evaluation
        final_metrics = self.evaluate() if self.eval_dataloader else {}

        # Save final model
        self.save_checkpoint("final_model")

        # Save routing history for convergence analysis
        if self.routing_history:
            self._save_routing_history()
            logger.info(f"Saved {len(self.routing_history)} routing snapshots")

        # Mark training as done
        self._mark_training_done()

        # Final push so TRAINING_DONE (and any straggler files) land on HF Hub
        # before this SLURM job exits — login-side chain_jobs.sh polls HF to
        # decide whether to resubmit, so the marker MUST be there.
        try:
            import sys
            from pathlib import Path as _P
            scripts_dir = _P(__file__).resolve().parents[3] / "scripts" / "transfer"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from ensimag_push import push_single_run
            push_single_run(_P(self.output_dir))
        except Exception as exc:
            logger.warning(f"final push skipped ({type(exc).__name__}): {exc}")

        return {
            "status": "completed",
            "final_train_loss": self.state.total_train_loss / max(self.state.num_train_steps, 1),
            "final_eval_metrics": final_metrics,
            "best_eval_loss": self.state.best_eval_loss,
            "total_steps": self.state.global_step,
            "training_time_minutes": elapsed_time / 60,
            "num_routing_snapshots": len(self.routing_history),
        }

    def _mark_training_done(self):
        """Create a marker file indicating training is complete."""
        done_file = self.output_dir / "TRAINING_DONE"
        with open(done_file, "w") as f:
            f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {self.num_epochs}\n")
            f.write(f"Total steps: {self.state.global_step}\n")
            f.write(f"Best eval loss: {self.state.best_eval_loss}\n")
        logger.info(f"Created {done_file}")

    def is_training_done(self) -> bool:
        """Check if training is already complete."""
        done_file = self.output_dir / "TRAINING_DONE"
        return done_file.exists()

    def save_checkpoint(self, name: str):
        """Save model checkpoint, then push older checkpoints to HF Hub.

        Push policy (best-effort, never crashes training):
        - The just-saved checkpoint stays local — it may be needed for resume.
        - All older `checkpoint-N/` (and `best_model`/`final_model`) are pushed
          to `Helain/gated-lora-experiments/<run>/<ckpt>/` and deleted locally.
        - On `TRAINING_DONE`, root files (`final_results.json`, `routing_history.json`,
          figures) are pushed and the run dir is reduced to the marker.

        Re-authentication on every call: compute nodes lose HF auth state
        between SLURM jobs, so the push helper logs in fresh from HF_TOKEN.
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save scheduler state
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # Save training state (including batch_idx for resume)
        state_dict = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "batch_idx": self.state.batch_idx,
            "best_eval_loss": self.state.best_eval_loss,
            "warmup_completed": self.state.warmup_completed,
            "total_train_loss": self.state.total_train_loss,
            "num_train_steps": self.state.num_train_steps,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Best-effort post-save push (older checkpoints → HF Hub, then delete).
        # Skipped silently if HF_TOKEN missing or push module unavailable.
        try:
            import sys
            from pathlib import Path as _P
            scripts_dir = _P(__file__).resolve().parents[3] / "scripts" / "transfer"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from ensimag_push import push_single_run
            push_single_run(_P(self.output_dir))
        except Exception as exc:
            logger.warning(f"post-save push skipped ({type(exc).__name__}): {exc}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        NOTE (2026-07 fix): the historical implementation restored optimizer/
        scheduler/training-state but silently SKIPPED the model weights for
        any model saved via ``save_pretrained`` (GatedLoRAModelV2 and PEFT
        baselines both are). Cross-job SLURM resume therefore restarted the
        adapters from scratch while reusing a stale optimizer state. This now
        loads the weights explicitly for both model families and refuses to
        resume when it cannot.
        """
        checkpoint_dir = Path(path)

        # Load model state
        if (checkpoint_dir / "model.pt").exists():
            state_dict = torch.load(
                checkpoint_dir / "model.pt", map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
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

        # Load optimizer state
        if (checkpoint_dir / "optimizer.pt").exists():
            self.optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pt", map_location=self.device)
            )
            logger.info(f"Loaded optimizer state")

        # Load scheduler state
        if self.scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
            self.scheduler.load_state_dict(
                torch.load(checkpoint_dir / "scheduler.pt", map_location=self.device)
            )
            logger.info(f"Loaded scheduler state")

        # Load training state
        if (checkpoint_dir / "training_state.json").exists():
            with open(checkpoint_dir / "training_state.json", "r") as f:
                state_dict = json.load(f)
                self.state.global_step = state_dict.get("global_step", 0)
                self.state.epoch = state_dict.get("epoch", 0)
                self.state.batch_idx = state_dict.get("batch_idx", 0)  # NEW
                self.state.best_eval_loss = state_dict.get("best_eval_loss", float("inf"))
                self.state.warmup_completed = state_dict.get("warmup_completed", False)
                self.state.total_train_loss = state_dict.get("total_train_loss", 0.0)  # NEW
                self.state.num_train_steps = state_dict.get("num_train_steps", 0)  # NEW

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        logger.info(f"  State: epoch={self.state.epoch}, batch_idx={self.state.batch_idx}, "
                   f"global_step={self.state.global_step}")


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

        if "gating" in name.lower() or "gate" in name.lower():
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

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # Create scheduler
    scheduler_name = getattr(config.training, "scheduler", "cosine").lower()
    warmup_ratio = getattr(config.training, "warmup_ratio", 0.1)
    warmup_steps = int(num_training_steps * warmup_ratio)

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
            T_max=num_training_steps - warmup_steps,
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
