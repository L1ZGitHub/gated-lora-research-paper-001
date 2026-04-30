#!/usr/bin/env python3
"""
Print formatted routing analysis summary from JSON files.

Usage:
    python print_routing_summary.py                    # All JSON files in current dir
    python print_routing_summary.py exp3_*.json       # Specific files
"""

import json
import sys
from pathlib import Path


def print_summary(results: dict, filename: str = None):
    """Print formatted routing summary like analyze_routing_standalone.py does."""

    experiment_name = results.get("experiment_name", filename or "unknown")
    num_experts = results.get("num_experts", 3)
    num_layers = results.get("num_layers", 32)
    gated_layers = results.get("gated_layers", None)

    task_usage = results.get("task_expert_usage", {})
    task_layer_usage = results.get("task_layer_expert_usage", {})
    metrics = results.get("specialization_metrics", {})
    per_layer_spec = results.get("per_layer_task_specialization", {})

    print("\n" + "=" * 70)
    print(f"ROUTING ANALYSIS SUMMARY: {experiment_name}")
    print("=" * 70)

    if gated_layers:
        print(f"\n[!] PARTIAL GATING: Only layers {gated_layers} are gated")

    # [1] Global task expert usage
    print(f"\n[1] GLOBAL Task Expert Usage (aggregated over all layers):")
    for task, usage in task_usage.items():
        usage_str = ", ".join([f"E{i}={u:.1%}" for i, u in enumerate(usage)])
        print(f"  {task}: {usage_str}")

    # [2] Global specialization metrics
    print(f"\n[2] GLOBAL Specialization Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # [3] Per-layer task specialization
    print(f"\n[3] PER-LAYER Task Specialization:")
    max_layer = per_layer_spec.get('max_specialization_layer', -1)
    if max_layer >= 0:
        print(f"  Max specialization at layer {max_layer} (score: {per_layer_spec['max_score']:.4f})")
        print(f"  Mean specialization score: {per_layer_spec.get('mean_score', 0):.4f}")
        print(f"  Top 5 layers: {per_layer_spec.get('top_5_layers', [])}")

        # [4] Detailed expert usage for top layers
        print(f"\n[4] DETAILED Expert Usage for Top Specializing Layers:")
        per_layer_scores = per_layer_spec.get('per_layer_scores', [])

        for layer_idx in per_layer_spec.get('top_5_layers', [])[:3]:
            score = per_layer_scores[layer_idx] if layer_idx < len(per_layer_scores) else 0
            print(f"\n  Layer {layer_idx} (spec_score: {score:.4f}):")
            for task in task_layer_usage.keys():
                if layer_idx < len(task_layer_usage[task]):
                    usage = task_layer_usage[task][layer_idx]
                    usage_str = ", ".join([f"E{i}={u:.1%}" for i, u in enumerate(usage)])
                    print(f"    {task}: {usage_str}")
    else:
        print("  No task specialization data available")

    print()


def main():
    # Get JSON files to process
    if len(sys.argv) > 1:
        json_files = [Path(f) for f in sys.argv[1:] if f.endswith('.json')]
    else:
        # All JSON files in current directory
        json_files = sorted(Path(".").glob("*_routing_analysis.json"))

    if not json_files:
        print("No routing analysis JSON files found.")
        print("Usage: python print_routing_summary.py [file1.json file2.json ...]")
        sys.exit(1)

    print(f"Processing {len(json_files)} files...")

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                results = json.load(f)
            print_summary(results, json_file.stem)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Print comparison table if multiple files
    if len(json_files) > 1:
        print("\n" + "=" * 85)
        print("COMPARISON TABLE")
        print("=" * 85)
        print(f"\n{'Experiment':<35} | {'Task Spec':<10} | {'Load Imb':<10} | {'Max Layer':<20}")
        print("-" * 85)

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    results = json.load(f)

                name = results.get("experiment_name", json_file.stem)
                metrics = results.get("specialization_metrics", {})
                per_layer = results.get("per_layer_task_specialization", {})

                task_spec = metrics.get("task_specialization_score", 0)
                load_imb = metrics.get("load_imbalance", 0)
                max_layer = per_layer.get("max_specialization_layer", -1)
                max_score = per_layer.get("max_score", 0)

                print(f"{name:<35} | {task_spec:<10.4f} | {load_imb:<10.4f} | L{max_layer} ({max_score:.3f})")
            except:
                pass

        print("=" * 85)


if __name__ == "__main__":
    main()
