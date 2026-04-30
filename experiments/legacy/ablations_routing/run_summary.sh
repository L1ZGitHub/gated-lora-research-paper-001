#!/bin/bash
#SBATCH --job-name=routing_summary
#SBATCH --output=summary_%j.out
#SBATCH --error=summary_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=4G

cd /home/helain/projects/gated-lora-research/ensicompute_ablations/routing_results

# Use python from environment
python3 print_routing_summary.py
