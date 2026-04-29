"""
Custom trainer for Gated LoRA research.

Features:
- HuggingFace Trainer compatible
- Custom training loop for gating flexibility
- Gradient accumulation
- Mixed precision (fp16/bf16)
- Gradient clipping
- Checkpoint saving (best + periodic)
- Early stopping
- VRAM monitoring
- OOM handling with batch size reduction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from tqdm.auto import tqdm
import json
from dataclasses import asdict

from .config import ExperimentConfig
from ..utils.logging import WandbLogger, MetricsTracker, setup_logging
from ..models.base_model import Phi2BaseModel
from ..models.gated_lora import GatedLoraModel

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    Custom trainer for LoRA and Gated LoRA models.

    Supports:
    - Gradient accumulation
    - Mixed precision training
    - Checkpoint management
    - Early stopping
    - VRAM monitoring
    - OOM recovery
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Experiment configuration
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            optimizer: Optional optimizer (will create if None)
            scheduler: Optional learning rate scheduler
            collate_fn: Optional collate function for dataloader
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn

        # Setup logging
        setup_logging(config.training.log_level)

        # Setup wandb
        self.wandb_logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name or config.experiment_name,
            config=config.to_dict(),
            tags=config.wandb.tags,
            notes=config.wandb.notes,
            mode=config.wandb.mode,
        )

        # Setup metrics tracker
        self.metrics_tracker = MetricsTracker()

        # Setup device
        self.device = model.device if hasattr(model, 'device') else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Setup mixed precision
        self.use_amp = config.training.fp16 or config.training.bf16
        self.scaler = GradScaler() if config.training.fp16 else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')  # For loss
        self.epochs_without_improvement = 0

        # OOM handling
        self.current_batch_size = config.training.batch_size
        self.oom_count = 0

        # Create dataloaders (must be done before scheduler)
        self.train_dataloader = self._create_dataloader(train_dataset, self.current_batch_size)
        self.eval_dataloader = self._create_dataloader(eval_dataset, self.current_batch_size) if eval_dataset else None

        # Setup optimizer and scheduler (after dataloaders)
        self.optimizer = optimizer if optimizer is not None else self._create_optimizer()
        self.scheduler = scheduler if scheduler is not None else self._create_scheduler()

        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Watch model with wandb (reduced frequency for performance)
        if self.wandb_logger.enabled:
            self.wandb_logger.watch_model(model, log="gradients", log_freq=config.training.logging_steps * 10)

        logger.info(f"Trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.current_batch_size}")
        logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {self.use_amp} (fp16={config.training.fp16}, bf16={config.training.bf16})")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        config = self.config.training

        # Get trainable parameters
        # model.model is the actual transformer (Phi2BaseModel is a wrapper)
        params = [p for p in self.model.model.parameters() if p.requires_grad]

        # Add gating network parameters if Gated LoRA
        if hasattr(self.model, 'gating_network'):
            params.extend([p for p in self.model.gating_network.parameters()])

        if config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        logger.info(f"Created optimizer: {config.optimizer}")
        return optimizer

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        config = self.config.training

        # Calculate total steps
        steps_per_epoch = len(self.train_dataloader) // config.gradient_accumulation_steps
        if config.max_steps > 0:
            total_steps = config.max_steps
        else:
            total_steps = steps_per_epoch * config.num_epochs

        if config.scheduler.lower() == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps,
            )
        elif config.scheduler.lower() == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
            )
        elif config.scheduler.lower() == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")

        if scheduler:
            logger.info(f"Created scheduler: {config.scheduler}")
        return scheduler

    def _create_dataloader(self, dataset: Optional[Dataset], batch_size: int) -> Optional[DataLoader]:
        """Create dataloader."""
        if dataset is None:
            return None

        config = self.config.training
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if dataset == self.train_dataset else False,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            collate_fn=self.collate_fn,
        )

    def _log_vram_usage(self):
        """Log VRAM usage and warn if exceeds limit."""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3

        if allocated > self.config.training.vram_limit_gb:
            logger.warning(f"VRAM usage ({allocated:.2f}GB) exceeds limit ({self.config.training.vram_limit_gb}GB)")

        return {"vram_allocated_gb": allocated, "vram_reserved_gb": reserved}

    def _save_checkpoint(self, metric_value: float, is_best: bool = False, filename: Optional[str] = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint-step-{self.global_step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Prepare checkpoint
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.model.state_dict(),  # Save inner model
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": asdict(self.config),
            "metric_value": metric_value,
        }

        # Add gating network state if Gated LoRA
        if hasattr(self.model, 'gating_network'):
            checkpoint["gating_network_state_dict"] = self.model.gating_network.state_dict()

        # Add scheduler state if exists
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

            # Log to wandb
            if self.wandb_logger.enabled:
                self.wandb_logger.log_artifact(
                    str(best_path),
                    artifact_type="model",
                    name="best_model",
                    aliases=["best"],
                )

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints according to save_total_limit."""
        config = self.config.training
        if config.save_total_limit <= 0:
            return

        # Get all checkpoint files
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint-step-*.pt")],
            key=lambda x: int(x.stem.split("-")[-1]),
        )

        # Remove old checkpoints
        if len(checkpoints) > config.save_total_limit:
            for checkpoint in checkpoints[:-config.save_total_limit]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")

    def _should_save(self) -> bool:
        """Check if should save checkpoint."""
        config = self.config.training
        if config.save_strategy == "no":
            return False
        elif config.save_strategy == "steps":
            return self.global_step % config.save_steps == 0
        elif config.save_strategy == "epoch":
            return True  # Called at end of epoch
        return False

    def _should_evaluate(self) -> bool:
        """Check if should evaluate."""
        config = self.config.training
        if config.eval_strategy == "no" or self.eval_dataloader is None:
            return False
        elif config.eval_strategy == "steps":
            return self.global_step % config.eval_steps == 0
        elif config.eval_strategy == "epoch":
            return True  # Called at end of epoch
        return False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        self.model.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with mixed precision
        # Determine dtype for autocast
        dtype = torch.bfloat16 if self.config.training.bf16 else (torch.float16 if self.config.training.fp16 else torch.float32)
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch["input_ids"]),
            )
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Prepare metrics
        loss_value = loss.item() * self.config.training.gradient_accumulation_steps

        # Check for NaN or inf loss
        if not torch.isfinite(torch.tensor(loss_value)):
            logger.error(f"Non-finite loss detected: {loss_value}")
            logger.error(f"Batch input_ids shape: {batch['input_ids'].shape}")
            logger.error(f"Batch input_ids sample: {batch['input_ids'][0][:20]}")
            if 'labels' in batch:
                logger.error(f"Labels sample: {batch['labels'][0][:20]}")
            raise ValueError(f"Non-finite loss detected: {loss_value}. Training cannot continue.")

        metrics = {"loss": loss_value}

        # Add gating-specific metrics if available
        if "lm_loss" in outputs:
            metrics["lm_loss"] = outputs["lm_loss"].item()
        if "load_balance_loss" in outputs:
            metrics["load_balance_loss"] = outputs["load_balance_loss"].item()

        return metrics

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        config = self.config.training

        # Unscale gradients if using mixed precision
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Clip gradients
        if config.max_grad_norm > 0:
            # Get all parameters from optimizer (includes model + gating network if present)
            params = []
            for group in self.optimizer.param_groups:
                params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(
                params,
                config.max_grad_norm,
            )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on evaluation dataset.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        self.model.model.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels", batch["input_ids"]),
                )

                total_loss += outputs["loss"].item()
                total_steps += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0

        return {"eval_loss": avg_loss}

    def train(self):
        """Main training loop."""
        config = self.config.training

        logger.info("Starting training...")
        logger.info(f"Num epochs: {config.num_epochs}")
        logger.info(f"Train dataset size: {len(self.train_dataset)}")

        try:
            for epoch in range(config.num_epochs):
                self.current_epoch = epoch
                logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")

                # Train one epoch
                self._train_epoch()

                # Evaluate at end of epoch
                if config.eval_strategy == "epoch" and self.eval_dataloader is not None:
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")

                    # Check for improvement
                    if self._check_improvement(eval_metrics.get("eval_loss", float('inf'))):
                        self._save_checkpoint(eval_metrics["eval_loss"], is_best=True)

                # Save at end of epoch
                if config.save_strategy == "epoch":
                    self._save_checkpoint(self.metrics_tracker.get_latest("loss") or 0.0)

                # Check early stopping
                if self._should_stop_early():
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Check max steps
                if config.max_steps > 0 and self.global_step >= config.max_steps:
                    logger.info(f"Reached max steps ({config.max_steps})")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM error: {e}")
                if config.oom_retry:
                    self._handle_oom()
            else:
                raise
        finally:
            # Final evaluation
            if self.eval_dataloader is not None:
                logger.info("\nFinal evaluation...")
                final_metrics = self.evaluate()
                self._log_metrics(final_metrics, prefix="final")

            # Log summary
            self._log_summary()

            # Finish wandb
            self.wandb_logger.finish()

            logger.info("Training completed!")

    def _train_epoch(self):
        """Train for one epoch."""
        config = self.config.training

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")

        accumulated_metrics = {}
        accumulation_count = 0

        for step, batch in enumerate(progress_bar):
            try:
                # Training step
                metrics = self.train_step(batch)

                # Accumulate metrics
                for key, value in metrics.items():
                    accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value
                accumulation_count += 1

                # Optimizer step after gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1

                    # Average accumulated metrics
                    avg_metrics = {
                        k: v / accumulation_count
                        for k, v in accumulated_metrics.items()
                    }
                    accumulated_metrics = {}
                    accumulation_count = 0

                    # Log metrics
                    if self.global_step % config.logging_steps == 0:
                        # Get VRAM metrics and combine with training metrics
                        vram_metrics = self._log_vram_usage()
                        if vram_metrics:
                            avg_metrics.update(vram_metrics)
                        self._log_metrics(avg_metrics)

                    # Update progress bar
                    progress_bar.set_postfix(avg_metrics)

                    # Evaluate
                    if self._should_evaluate():
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, prefix="eval")

                        # Check for improvement
                        if self._check_improvement(eval_metrics.get("eval_loss", float('inf'))):
                            self._save_checkpoint(eval_metrics["eval_loss"], is_best=True)

                    # Save checkpoint
                    if self._should_save():
                        current_loss = self.metrics_tracker.get_latest("loss") or 0.0
                        self._save_checkpoint(current_loss)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM during training step: {e}")
                    if config.oom_retry:
                        self._handle_oom()
                        break  # Restart epoch with smaller batch size
                    else:
                        raise
                else:
                    raise

    def _handle_oom(self):
        """Handle out-of-memory error by reducing batch size."""
        config = self.config.training

        logger.warning("Handling OOM error...")

        # Clear cache
        torch.cuda.empty_cache()

        # Reduce batch size
        new_batch_size = max(self.current_batch_size // 2, config.min_batch_size)

        if new_batch_size < config.min_batch_size:
            logger.error(f"Batch size {new_batch_size} is below minimum {config.min_batch_size}. Cannot continue.")
            raise RuntimeError("Cannot reduce batch size further")

        logger.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size}")

        self.current_batch_size = new_batch_size
        self.oom_count += 1

        # Recreate dataloaders
        self.train_dataloader = self._create_dataloader(self.train_dataset, new_batch_size)
        if self.eval_dataset is not None:
            self.eval_dataloader = self._create_dataloader(self.eval_dataset, new_batch_size)

        logger.info("Restarting training with smaller batch size...")

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to wandb and tracker."""
        # Add prefix
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Add learning rate
        if self.optimizer is not None:
            prefixed_metrics["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Log to wandb
        self.wandb_logger.log(prefixed_metrics, step=self.global_step)

        # Update tracker
        self.metrics_tracker.update(prefixed_metrics, self.global_step)

        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} - {metrics_str}")

    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best."""
        improved = current_metric < self.best_metric

        if improved:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return improved

    def _should_stop_early(self) -> bool:
        """Check if should stop training early."""
        config = self.config.training
        return (
            config.early_stopping_patience > 0
            and self.epochs_without_improvement >= config.early_stopping_patience
        )

    def _log_summary(self):
        """Log training summary."""
        summary = self.metrics_tracker.get_summary()
        summary["total_steps"] = self.global_step
        summary["total_epochs"] = self.current_epoch + 1
        summary["oom_count"] = self.oom_count
        summary["final_batch_size"] = self.current_batch_size

        self.wandb_logger.log_summary(summary)

        logger.info("\nTraining Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    print("Trainer module - use train.py script to run training")
