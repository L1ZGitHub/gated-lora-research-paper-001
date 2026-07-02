"""
Routing Analysis and Visualization Tools for Gated LoRA.

Provides:
1. Expert usage analysis per layer and per task
2. Routing entropy visualization
3. Token-level routing heatmaps
4. Expert specialization analysis
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import logging
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RoutingAnalyzer:
    """
    Analyzes routing patterns from Gated LoRA models.

    Collects and aggregates routing statistics for visualization and analysis.
    """

    def __init__(self, num_experts: int, num_layers: int):
        self.num_experts = num_experts
        self.num_layers = num_layers

        # Storage for routing data
        self.expert_usage_per_layer: List[List[float]] = [[] for _ in range(num_layers)]
        self.entropy_per_layer: List[List[float]] = [[] for _ in range(num_layers)]
        self.expert_usage_per_task: Dict[str, List[List[float]]] = defaultdict(lambda: [[] for _ in range(num_experts)])
        self.routing_decisions: List[Dict[str, Any]] = []

        # NEW: Per-layer, per-task routing (for fine-grained analysis)
        # Structure: task -> layer -> list of [num_experts] arrays
        self.expert_usage_per_task_per_layer: Dict[str, List[List[np.ndarray]]] = defaultdict(
            lambda: [[] for _ in range(num_layers)]
        )

        # Token-level routing (for visualization)
        self.token_routings: List[Dict[str, Any]] = []

    def record_routing(
        self,
        gate_weights: torch.Tensor,
        layer_idx: int,
        task_id: Optional[str] = None,
        tokens: Optional[List[str]] = None,
    ):
        """
        Record routing decision for analysis.

        Args:
            gate_weights: [batch, seq_len, num_experts] routing weights
            layer_idx: Which transformer layer
            task_id: Optional task identifier for multi-task analysis
            tokens: Optional list of tokens for visualization
        """
        # Move to CPU and convert to numpy (float32 since numpy doesn't support bfloat16)
        weights = gate_weights.detach().cpu().float().numpy()

        # Expert usage for this batch
        batch_expert_usage = weights.mean(axis=(0, 1))  # [num_experts]
        self.expert_usage_per_layer[layer_idx].append(batch_expert_usage)

        # Entropy
        eps = 1e-8
        entropy = -np.sum(weights * np.log(weights + eps), axis=-1).mean()
        self.entropy_per_layer[layer_idx].append(entropy)

        # Per-task usage (aggregated)
        if task_id is not None:
            for expert_idx in range(self.num_experts):
                self.expert_usage_per_task[task_id][expert_idx].append(
                    batch_expert_usage[expert_idx]
                )
            # NEW: Per-task, per-layer usage
            self.expert_usage_per_task_per_layer[task_id][layer_idx].append(
                batch_expert_usage
            )

        # Token-level routing (store first example for visualization)
        if tokens is not None and len(self.token_routings) < 100:
            self.token_routings.append({
                "tokens": tokens[:50],  # Limit tokens
                "weights": weights[0, :50, :].tolist(),  # First batch item
                "layer_idx": layer_idx,
                "task_id": task_id,
            })

    def get_layer_expert_usage(self) -> np.ndarray:
        """
        Get aggregated expert usage per layer.

        Returns:
            Array of shape [num_layers, num_experts]
        """
        usage = np.zeros((self.num_layers, self.num_experts))

        for layer_idx in range(self.num_layers):
            if self.expert_usage_per_layer[layer_idx]:
                usage[layer_idx] = np.mean(self.expert_usage_per_layer[layer_idx], axis=0)

        return usage

    def get_layer_entropy(self) -> np.ndarray:
        """
        Get average routing entropy per layer.

        Returns:
            Array of shape [num_layers]
        """
        entropy = np.zeros(self.num_layers)

        for layer_idx in range(self.num_layers):
            if self.entropy_per_layer[layer_idx]:
                entropy[layer_idx] = np.mean(self.entropy_per_layer[layer_idx])

        return entropy

    def get_task_expert_usage(self) -> Dict[str, np.ndarray]:
        """
        Get expert usage breakdown per task.

        Returns:
            Dict mapping task_id -> [num_experts] usage array
        """
        result = {}

        for task_id, expert_lists in self.expert_usage_per_task.items():
            task_usage = np.zeros(self.num_experts)
            for expert_idx, values in enumerate(expert_lists):
                if values:
                    task_usage[expert_idx] = np.mean(values)
            result[task_id] = task_usage

        return result

    def get_task_layer_expert_usage(self) -> Dict[str, np.ndarray]:
        """
        Get expert usage per task AND per layer.

        Returns:
            Dict mapping task_id -> [num_layers, num_experts] usage array
        """
        result = {}

        for task_id, layer_lists in self.expert_usage_per_task_per_layer.items():
            task_layer_usage = np.zeros((self.num_layers, self.num_experts))
            for layer_idx, usage_list in enumerate(layer_lists):
                if usage_list:
                    task_layer_usage[layer_idx] = np.mean(usage_list, axis=0)
            result[task_id] = task_layer_usage

        return result

    def compute_per_layer_task_specialization(self) -> Dict[str, Any]:
        """
        Compute task specialization score for EACH layer.

        This shows which layers differentiate most between tasks.

        Returns:
            Dict with per-layer specialization scores and summary
        """
        task_layer_usage = self.get_task_layer_expert_usage()

        if len(task_layer_usage) < 2:
            return {"per_layer_scores": [], "max_specialization_layer": -1, "max_score": 0.0}

        tasks = list(task_layer_usage.keys())
        per_layer_scores = []

        for layer_idx in range(self.num_layers):
            # Get usage for this layer across all tasks
            layer_usages = []
            for task in tasks:
                usage = task_layer_usage[task][layer_idx]
                # Normalize
                usage = usage / (usage.sum() + 1e-8)
                layer_usages.append(usage)

            # Compute pairwise distances for this layer
            total_distance = 0.0
            num_pairs = 0
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    distance = np.linalg.norm(layer_usages[i] - layer_usages[j])
                    total_distance += distance
                    num_pairs += 1

            if num_pairs > 0:
                avg_distance = total_distance / num_pairs
                score = avg_distance / np.sqrt(2)  # Normalize
            else:
                score = 0.0

            per_layer_scores.append(score)

        per_layer_scores = np.array(per_layer_scores)
        max_layer = int(np.argmax(per_layer_scores))

        return {
            "per_layer_scores": per_layer_scores.tolist(),
            "max_specialization_layer": max_layer,
            "max_score": float(per_layer_scores[max_layer]),
            "mean_score": float(per_layer_scores.mean()),
            "top_5_layers": np.argsort(per_layer_scores)[-5:][::-1].tolist(),
        }

    def compute_expert_specialization(self) -> Dict[str, float]:
        """
        Compute metrics about expert specialization.

        Returns:
            Dict with specialization metrics
        """
        layer_usage = self.get_layer_expert_usage()

        # Load imbalance (std of usage across experts)
        load_imbalance = np.std(layer_usage, axis=1).mean()

        # Expert diversity (how different are experts across layers)
        expert_variance = np.var(layer_usage, axis=0).mean()

        # Dominant expert ratio (how often does one expert dominate)
        dominant_ratios = layer_usage.max(axis=1) / (layer_usage.sum(axis=1) + 1e-8)
        avg_dominance = dominant_ratios.mean()

        # Task specialization score: measures how much expert usage differs between tasks
        task_specialization = self.compute_task_specialization_score()

        return {
            "load_imbalance": float(load_imbalance),
            "expert_variance": float(expert_variance),
            "avg_dominance": float(avg_dominance),
            "max_dominance": float(dominant_ratios.max()),
            "task_specialization_score": float(task_specialization),
        }

    def compute_task_specialization_score(self) -> float:
        """
        Compute task specialization score.

        Measures how differently the experts are used across tasks.
        High score = experts are used differently for different tasks (good!)
        Low score = same expert usage regardless of task (bad!)

        Returns:
            Specialization score (0 = no specialization, higher = more specialization)
        """
        task_usage = self.get_task_expert_usage()

        if len(task_usage) < 2:
            # Need at least 2 tasks to measure specialization
            return 0.0

        # Convert to matrix: [num_tasks, num_experts]
        tasks = list(task_usage.keys())
        usage_matrix = np.array([task_usage[t] for t in tasks])

        # Normalize each task's usage to sum to 1
        usage_matrix = usage_matrix / (usage_matrix.sum(axis=1, keepdims=True) + 1e-8)

        # Compute pairwise distances between task usage patterns
        # Using Jensen-Shannon divergence would be ideal, but L2 distance works too
        total_distance = 0.0
        num_pairs = 0

        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                # L2 distance between usage patterns
                distance = np.linalg.norm(usage_matrix[i] - usage_matrix[j])
                total_distance += distance
                num_pairs += 1

        if num_pairs == 0:
            return 0.0

        # Average distance, normalized by max possible distance
        # Max distance for normalized vectors is sqrt(2)
        avg_distance = total_distance / num_pairs
        max_distance = np.sqrt(2)  # For normalized probability vectors

        return avg_distance / max_distance

    def save_analysis(self, path: str):
        """Save analysis results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get per-layer task specialization
        per_layer_spec = self.compute_per_layer_task_specialization()

        # Get task-layer usage matrix
        task_layer_usage = self.get_task_layer_expert_usage()

        analysis = {
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "layer_expert_usage": self.get_layer_expert_usage().tolist(),
            "layer_entropy": self.get_layer_entropy().tolist(),
            "task_expert_usage": {k: v.tolist() for k, v in self.get_task_expert_usage().items()},
            "task_layer_expert_usage": {k: v.tolist() for k, v in task_layer_usage.items()},
            "specialization_metrics": self.compute_expert_specialization(),
            "per_layer_task_specialization": per_layer_spec,
        }

        with open(path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Saved routing analysis to {path}")

    def clear(self):
        """Clear all recorded data."""
        self.expert_usage_per_layer = [[] for _ in range(self.num_layers)]
        self.entropy_per_layer = [[] for _ in range(self.num_layers)]
        self.expert_usage_per_task = defaultdict(lambda: [[] for _ in range(self.num_experts)])
        self.routing_decisions = []
        self.token_routings = []


class RoutingVisualizer:
    """
    Creates visualizations for routing analysis.
    """

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - visualization disabled")

    def plot_layer_expert_heatmap(
        self,
        layer_usage: np.ndarray,
        expert_names: Optional[List[str]] = None,
        title: str = "Expert Usage per Layer",
        filename: str = "layer_expert_heatmap.png",
    ):
        """
        Create heatmap showing expert usage across layers.

        Args:
            layer_usage: [num_layers, num_experts] array
            expert_names: Optional names for experts
            title: Plot title
            filename: Output filename
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        num_layers, num_experts = layer_usage.shape

        if expert_names is None:
            expert_names = [f"Expert {i}" for i in range(num_experts)]

        fig, ax = plt.subplots(figsize=(10, 14))

        if SEABORN_AVAILABLE:
            sns.heatmap(
                layer_usage,
                ax=ax,
                cmap="YlOrRd",
                annot=False,
                xticklabels=expert_names,
                yticklabels=[f"Layer {i}" for i in range(num_layers)],
                cbar_kws={"label": "Usage Probability"},
            )
        else:
            im = ax.imshow(layer_usage, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(num_experts))
            ax.set_xticklabels(expert_names)
            ax.set_yticks(range(0, num_layers, 4))
            ax.set_yticklabels([f"Layer {i}" for i in range(0, num_layers, 4)])
            plt.colorbar(im, ax=ax, label="Usage Probability")

        ax.set_xlabel("Expert")
        ax.set_ylabel("Transformer Layer")
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved {filename}")

    def plot_entropy_per_layer(
        self,
        entropy: np.ndarray,
        title: str = "Routing Entropy per Layer",
        filename: str = "entropy_per_layer.png",
        num_experts: int = 3,
    ):
        """
        Plot routing entropy across layers.

        Args:
            entropy: [num_layers] array
            title: Plot title
            filename: Output filename
            num_experts: expert count, used for the max-entropy reference line
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        num_layers = len(entropy)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.bar(range(num_layers), entropy, color="steelblue", alpha=0.7)
        ax.axhline(y=entropy.mean(), color="red", linestyle="--", label=f"Mean: {entropy.mean():.3f}")

        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Routing Entropy")
        ax.set_title(title)
        ax.legend()

        # Add max entropy reference line.
        # Upper bound of routing entropy = log(num_experts): uniform routing
        # across experts. (Was log(num_layers) — wrong reference line.)
        max_entropy = np.log(max(num_experts, 2))
        ax.axhline(y=max_entropy, color="gray", linestyle=":", alpha=0.5, label="Max Entropy")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved {filename}")

    def plot_task_expert_usage(
        self,
        task_usage: Dict[str, np.ndarray],
        expert_names: Optional[List[str]] = None,
        title: str = "Expert Usage by Task",
        filename: str = "task_expert_usage.png",
    ):
        """
        Plot expert usage breakdown per task.

        Args:
            task_usage: Dict mapping task_id -> [num_experts] usage
            expert_names: Optional names for experts
            title: Plot title
            filename: Output filename
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        tasks = list(task_usage.keys())
        num_experts = len(list(task_usage.values())[0])

        if expert_names is None:
            expert_names = [f"Expert {i}" for i in range(num_experts)]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(tasks))
        width = 0.8 / num_experts

        colors = plt.cm.Set2(np.linspace(0, 1, num_experts))

        for i, expert_name in enumerate(expert_names):
            values = [task_usage[task][i] for task in tasks]
            ax.bar(x + i * width, values, width, label=expert_name, color=colors[i])

        ax.set_xlabel("Task")
        ax.set_ylabel("Expert Usage")
        ax.set_title(title)
        ax.set_xticks(x + width * (num_experts - 1) / 2)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved {filename}")

    def plot_token_routing_heatmap(
        self,
        tokens: List[str],
        weights: np.ndarray,
        expert_names: Optional[List[str]] = None,
        title: str = "Token-Level Routing",
        filename: str = "token_routing.png",
    ):
        """
        Create heatmap showing routing weights for each token.

        Args:
            tokens: List of token strings
            weights: [seq_len, num_experts] array
            expert_names: Optional names for experts
            title: Plot title
            filename: Output filename
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        seq_len, num_experts = weights.shape

        if expert_names is None:
            expert_names = [f"Expert {i}" for i in range(num_experts)]

        # Limit display tokens
        max_tokens = min(50, seq_len)
        tokens = tokens[:max_tokens]
        weights = weights[:max_tokens]

        fig, ax = plt.subplots(figsize=(15, 5))

        if SEABORN_AVAILABLE:
            sns.heatmap(
                weights.T,
                ax=ax,
                cmap="Blues",
                xticklabels=tokens,
                yticklabels=expert_names,
                cbar_kws={"label": "Routing Weight"},
            )
        else:
            im = ax.imshow(weights.T, aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(num_experts))
            ax.set_yticklabels(expert_names)
            plt.colorbar(im, ax=ax, label="Routing Weight")

        ax.set_xlabel("Token")
        ax.set_ylabel("Expert")
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved {filename}")

    def plot_training_routing_evolution(
        self,
        routing_history: List[Dict[str, float]],
        title: str = "Routing Statistics During Training",
        filename: str = "routing_evolution.png",
    ):
        """
        Plot how routing statistics evolve during training.

        Args:
            routing_history: List of routing stat dicts with step info
            title: Plot title
            filename: Output filename
        """
        if not MATPLOTLIB_AVAILABLE or not routing_history:
            return

        steps = [h.get("step", i) for i, h in enumerate(routing_history)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Expert usage over time
        ax1 = axes[0, 0]
        for expert_idx in range(3):  # Assume 3 experts
            key = f"avg_expert_{expert_idx}_usage"
            if key in routing_history[0]:
                values = [h.get(key, 0) for h in routing_history]
                ax1.plot(steps, values, label=f"Expert {expert_idx}")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Usage")
        ax1.set_title("Expert Usage Over Time")
        ax1.legend()

        # Entropy over time
        ax2 = axes[0, 1]
        if "avg_routing_entropy" in routing_history[0]:
            entropy = [h.get("avg_routing_entropy", 0) for h in routing_history]
            ax2.plot(steps, entropy, color="purple")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Routing Entropy Over Time")

        # Load imbalance over time
        ax3 = axes[1, 0]
        if "avg_load_imbalance" in routing_history[0]:
            imbalance = [h.get("avg_load_imbalance", 0) for h in routing_history]
            ax3.plot(steps, imbalance, color="orange")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Load Imbalance")
        ax3.set_title("Load Imbalance Over Time")

        # Top-1 dominance over time
        ax4 = axes[1, 1]
        if "avg_top1_dominance" in routing_history[0]:
            dominance = [h.get("avg_top1_dominance", 0) for h in routing_history]
            ax4.plot(steps, dominance, color="green")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Top-1 Dominance")
        ax4.set_title("Top-1 Dominance Over Time")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved {filename}")

    def create_full_report(
        self,
        analyzer: RoutingAnalyzer,
        experiment_name: str = "gated_lora",
    ):
        """
        Create a full visualization report from analyzer data.

        Args:
            analyzer: RoutingAnalyzer with collected data
            experiment_name: Name for the experiment
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping visualization report")
            return

        # Expert names based on the ACTUAL configured ranks when available
        # (was hardcoded [8, 16, 32] — wrong labels for e.g. 2-expert runs).
        ranks = getattr(analyzer, "expert_ranks", None)
        if ranks and len(ranks) == analyzer.num_experts:
            expert_names = [f"Expert {i}\n(r={ranks[i]})" for i in range(analyzer.num_experts)]
        else:
            expert_names = [f"Expert {i}" for i in range(analyzer.num_experts)]

        # 1. Layer-expert heatmap
        layer_usage = analyzer.get_layer_expert_usage()
        self.plot_layer_expert_heatmap(
            layer_usage,
            expert_names=expert_names,
            title=f"{experiment_name}: Expert Usage per Layer",
            filename=f"{experiment_name}_layer_expert_heatmap.png",
        )

        # 2. Entropy per layer
        entropy = analyzer.get_layer_entropy()
        self.plot_entropy_per_layer(
            entropy,
            title=f"{experiment_name}: Routing Entropy per Layer",
            filename=f"{experiment_name}_entropy_per_layer.png",
            num_experts=analyzer.num_experts,
        )

        # 3. Task-expert usage (if available)
        task_usage = analyzer.get_task_expert_usage()
        if task_usage:
            self.plot_task_expert_usage(
                task_usage,
                expert_names=[f"Expert {i}" for i in range(analyzer.num_experts)],
                title=f"{experiment_name}: Expert Usage by Task",
                filename=f"{experiment_name}_task_expert_usage.png",
            )

        # 4. Token routing examples
        if analyzer.token_routings:
            for i, routing in enumerate(analyzer.token_routings[:3]):  # First 3 examples
                self.plot_token_routing_heatmap(
                    routing["tokens"],
                    np.array(routing["weights"]),
                    expert_names=[f"Expert {j}" for j in range(analyzer.num_experts)],
                    title=f"{experiment_name}: Token Routing (Example {i+1})",
                    filename=f"{experiment_name}_token_routing_{i+1}.png",
                )

        # Save analysis JSON
        analyzer.save_analysis(self.output_dir / f"{experiment_name}_analysis.json")

        logger.info(f"Created full visualization report for {experiment_name}")


def analyze_model_routing(
    model,
    dataloader,
    tokenizer,
    num_batches: int = 10,
    output_dir: str = "./visualizations",
    experiment_name: str = "gated_lora",
) -> Dict[str, Any]:
    """
    Run routing analysis on a trained gated LoRA model.

    Args:
        model: Trained GatedLoRAModel
        dataloader: DataLoader with evaluation data
        tokenizer: Tokenizer for decoding tokens
        num_batches: Number of batches to analyze
        output_dir: Where to save visualizations
        experiment_name: Name for the experiment

    Returns:
        Dict with analysis results
    """
    device = next(model.parameters()).device

    # Determine model structure
    num_experts = getattr(model, "num_experts", 3)
    num_layers = getattr(model, "num_layers", 32)

    analyzer = RoutingAnalyzer(num_experts=num_experts, num_layers=num_layers)
    visualizer = RoutingVisualizer(output_dir=output_dir)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass to get routing info
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_routing_info=True,
            )

            # Extract routing info
            if isinstance(outputs, dict) and "routing_info" in outputs:
                routing_info = outputs["routing_info"]

                # Get tasks for all samples in batch
                batch_tasks = batch.get("task", None)

                # Decode tokens for visualization (first sample)
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

                # Check for per_layer_info (from GatedLoRAModelV2)
                if "per_layer_info" in routing_info:
                    per_layer_info = routing_info["per_layer_info"]
                    # Record routing for each layer
                    for layer_idx, layer_info in per_layer_info.items():
                        if "gate_weights" in layer_info:
                            gate_weights = layer_info["gate_weights"]
                            layer_idx_int = int(layer_idx) if isinstance(layer_idx, str) else layer_idx

                            # Record per-task routing (for each sample in batch)
                            if batch_tasks is not None:
                                for sample_idx, task_id in enumerate(batch_tasks):
                                    # Extract this sample's gate weights
                                    sample_weights = gate_weights[sample_idx:sample_idx+1]
                                    analyzer.record_routing(
                                        sample_weights,
                                        layer_idx=layer_idx_int,
                                        task_id=task_id,
                                        tokens=tokens if (layer_idx_int == num_layers // 2 and sample_idx == 0) else None,
                                    )
                            else:
                                # No task info, record whole batch
                                analyzer.record_routing(
                                    gate_weights,
                                    layer_idx=layer_idx_int,
                                    task_id=None,
                                    tokens=tokens if layer_idx_int == num_layers // 2 else None,
                                )
                # Fallback: direct gate_weights
                elif "gate_weights" in routing_info:
                    gate_weights = routing_info["gate_weights"]
                    if batch_tasks is not None:
                        for sample_idx, task_id in enumerate(batch_tasks):
                            sample_weights = gate_weights[sample_idx:sample_idx+1]
                            analyzer.record_routing(
                                sample_weights,
                                layer_idx=num_layers // 2,
                                task_id=task_id,
                                tokens=tokens if sample_idx == 0 else None,
                            )
                    else:
                        analyzer.record_routing(
                            gate_weights,
                            layer_idx=num_layers // 2,
                            task_id=None,
                            tokens=tokens,
                        )

    # Create visualizations
    visualizer.create_full_report(analyzer, experiment_name=experiment_name)

    # Return summary metrics
    return analyzer.compute_expert_specialization()


if __name__ == "__main__":
    # Test the analyzer and visualizer
    print("Testing RoutingAnalyzer...")

    analyzer = RoutingAnalyzer(num_experts=3, num_layers=32)

    # Simulate some routing data
    for _ in range(100):
        # Random routing weights
        weights = torch.softmax(torch.randn(2, 128, 3), dim=-1)

        for layer_idx in range(32):
            analyzer.record_routing(
                weights,
                layer_idx=layer_idx,
                task_id=np.random.choice(["squad", "imdb", "wikitext"]),
            )

    # Get analysis
    print("\nLayer-expert usage shape:", analyzer.get_layer_expert_usage().shape)
    print("Layer entropy shape:", analyzer.get_layer_entropy().shape)
    print("Task usage:", list(analyzer.get_task_expert_usage().keys()))
    print("Specialization metrics:", analyzer.compute_expert_specialization())

    # Test visualization (if matplotlib available)
    if MATPLOTLIB_AVAILABLE:
        print("\nCreating visualizations...")
        visualizer = RoutingVisualizer(output_dir="./test_visualizations")
        visualizer.create_full_report(analyzer, experiment_name="test")
        print("Done!")
    else:
        print("matplotlib not available - skipping visualization test")
