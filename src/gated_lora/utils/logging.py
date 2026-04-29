"""
Logging utilities for training with wandb integration.
"""

import wandb
import logging
import sys
from typing import Dict, Any, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Wrapper for Weights & Biases logging.

    Handles initialization, logging, and cleanup of wandb runs.
    """

    def __init__(
        self,
        project: str = "gated-lora-research",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        dir: str = "./wandb",
        mode: str = "online",  # "online", "offline", or "disabled"
        resume: Optional[str] = None,
    ):
        """
        Initialize wandb logger.

        Args:
            project: wandb project name
            name: run name
            config: configuration dictionary
            tags: list of tags
            notes: run notes
            dir: directory to save wandb files
            mode: wandb mode (online, offline, disabled)
            resume: resume mode ("allow", "must", "never", or run_id)
        """
        self.project = project
        self.name = name
        self.mode = mode
        self.enabled = mode != "disabled"

        if self.enabled:
            # Initialize wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                dir=dir,
                mode=mode,
                resume=resume,
            )

            logger.info(f"Initialized wandb run: {self.run.name}")
            logger.info(f"wandb URL: {self.run.url}")
        else:
            self.run = None
            logger.info("wandb logging disabled")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to wandb.

        Args:
            metrics: dictionary of metrics to log
            step: optional step number
            commit: whether to commit immediately
        """
        if not self.enabled:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
        aliases: Optional[list] = None,
    ):
        """
        Log an artifact (model, dataset, etc.) to wandb.

        Args:
            artifact_path: path to artifact
            artifact_type: type of artifact
            name: artifact name
            aliases: list of aliases (e.g., ["latest", "best"])
        """
        if not self.enabled:
            return

        try:
            if name is None:
                name = Path(artifact_path).name

            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_dir(artifact_path)

            if aliases:
                self.run.log_artifact(artifact, aliases=aliases)
            else:
                self.run.log_artifact(artifact)

            logger.info(f"Logged artifact: {name}")
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model parameters and gradients.

        Args:
            model: model to watch
            log: what to log ("gradients", "parameters", "all", or None)
            log_freq: logging frequency
        """
        if not self.enabled:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
            logger.info(f"Watching model with log={log}, freq={log_freq}")
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")

    def log_summary(self, summary: Dict[str, Any]):
        """
        Log summary statistics.

        Args:
            summary: dictionary of summary statistics
        """
        if not self.enabled:
            return

        try:
            for key, value in summary.items():
                wandb.run.summary[key] = value
            logger.info("Logged summary statistics")
        except Exception as e:
            logger.warning(f"Failed to log summary: {e}")

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
                logger.info("Finished wandb run")
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class MetricsTracker:
    """
    Track metrics during training.

    Keeps history of metrics and computes statistics.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {}
        self.best_metrics = {}

    def update(self, metrics: Dict[str, float], step: int):
        """
        Update metrics.

        Args:
            metrics: dictionary of metrics
            step: current step
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))

            # Track best metrics
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                # For loss, lower is better; for accuracy, higher is better
                if "loss" in key.lower():
                    if value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                else:
                    if value > self.best_metrics[key]:
                        self.best_metrics[key] = value

    def get_history(self, key: str) -> list:
        """Get history for a specific metric."""
        return self.history.get(key, [])

    def get_best(self, key: str) -> Optional[float]:
        """Get best value for a specific metric."""
        return self.best_metrics.get(key)

    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a specific metric."""
        history = self.history.get(key, [])
        return history[-1][1] if history else None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for key in self.history.keys():
            summary[f"{key}_best"] = self.get_best(key)
            summary[f"{key}_latest"] = self.get_latest(key)
        return summary


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Logs to stdout instead of stderr for better integration with SLURM job output.

    Args:
        log_level: logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: optional log file path
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create stdout handler (instead of default stderr)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(getattr(logging, log_level.upper()))
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


if __name__ == "__main__":
    # Test logging
    setup_logging("INFO")

    # Test wandb logger
    with WandbLogger(project="test", mode="disabled") as wandb_logger:
        wandb_logger.log({"test_metric": 0.5}, step=1)
        print("Logging test completed")
