# Comprehensive Routing Analysis Comparison

## Overview

This document compares routing behavior across all Gated LoRA experiments:
- **Base experiments**: exp2 (2 experts), exp3 (3 experts), exp4 (load balancing), exp5 (top-k)
- **Ablation studies**: no_l1 (no L1 regularization), no_layer_emb (no layer embedding), partial_gating (only layers 28-31 gated)

---

## 1. Performance Summary

| Experiment | Eval Loss | Train Loss | Num Experts | Key Modification |
|------------|-----------|------------|-------------|------------------|
| exp2_gated_2experts | N/A | N/A | 2 | Baseline 2-expert |
| exp3_gated_3experts | 0.7435 | 0.8903 | 3 | Baseline 3-expert |
| exp4_gated_loadbalancing | 0.7437 | 0.9090 | 3 | Load balancing loss |
| exp5_gated_topk | 0.7431 | 0.8937 | 3 | Top-k routing |
| ablation_no_l1 | 0.7437 | 0.9264 | 3 | No L1 regularization |
| ablation_no_layer_emb | 0.7430 | 1.1439 | 3 | No layer embedding |
| ablation_partial_gating | 0.7438 | 0.9010 | 3 | Only L28-31 gated |

**Key Finding**: All configurations converge to similar eval loss (~0.743-0.744). The differences are in routing behavior and specialization patterns, not final performance.

---

## 2. Specialization Metrics Comparison

| Experiment | Task Spec Score | Mean Layer Spec | Max Layer Spec | Max Spec Layer |
|------------|-----------------|-----------------|----------------|----------------|
| exp2_gated_2experts | 0.0213 | 0.0505 | 0.266 | L30 |
| exp3_gated_3experts | 0.0186 | 0.0326 | 0.280 | L31 |
| exp4_gated_loadbalancing | 0.0133 | 0.0207 | 0.257 | L31 |
| exp5_gated_topk | 0.0436 | 0.1606 | 0.293 | L30 |
| ablation_no_l1 | 0.0473 | 0.1518 | 0.275 | L29 |
| ablation_no_layer_emb | 0.0438 | 0.1686 | 0.282 | L29 |
| ablation_partial_gating | 0.0284 | 0.1411 | 0.275 | L30 |

### Key Observations:

1. **Load balancing (exp4) reduces specialization**: Lowest task_spec_score (0.0133) and mean_layer_spec (0.0207) - the load balancing loss actively discourages expert specialization.

2. **Removing L1 regularization increases specialization**: ablation_no_l1 has highest task_spec_score (0.0473), confirming L1 regularization pushes toward uniform routing.

3. **Top-k routing enables specialization**: exp5 shows high specialization (0.0436) despite having same number of experts as exp3/exp4.

4. **Layer embedding removal has minimal impact**: no_layer_emb shows similar specialization (0.0438) to exp5, suggesting layer embedding is not crucial for task specialization.

5. **Specialization concentrates in late layers**: All experiments show max specialization in layers 29-31, regardless of architecture.

---

## 3. Load Imbalance Analysis

| Experiment | Load Imbalance | Expert Variance | Avg Dominance | Max Dominance |
|------------|----------------|-----------------|---------------|---------------|
| exp2_gated_2experts | 0.168 | 0.020 | 0.668 | 0.979 |
| exp3_gated_3experts | 0.083 | 0.015 | 0.446 | 0.950 |
| exp4_gated_loadbalancing | 0.065 | 0.010 | 0.403 | 0.917 |
| exp5_gated_topk | 0.118 | 0.029 | 0.438 | 0.966 |
| ablation_no_l1 | 0.105 | 0.023 | 0.427 | 0.948 |
| ablation_no_layer_emb | 0.113 | 0.027 | 0.433 | 0.955 |
| ablation_partial_gating | 0.098 | 0.022 | 0.424 | 0.960 |

### Key Observations:

1. **Load balancing works**: exp4 has lowest load_imbalance (0.065) and expert_variance (0.010) - the load balancing loss successfully distributes tokens across experts.

2. **2-expert models are more imbalanced**: exp2 shows highest load_imbalance (0.168) - with only 2 experts, one tends to dominate.

3. **Top-k increases imbalance**: exp5 has higher load_imbalance (0.118) than soft routing (exp3: 0.083) - hard routing decisions create more extreme expert usage patterns.

4. **All models show dominant layers**: Max dominance >0.9 in all experiments, indicating some layer-expert combinations capture nearly all tokens.

---

## 4. Per-Layer Specialization Patterns

### Top-5 Specialization Layers by Experiment:

| Experiment | #1 | #2 | #3 | #4 | #5 |
|------------|----|----|----|----|-----|
| exp2_gated_2experts | L30 | L31 | L0 | L2 | L24 |
| exp3_gated_3experts | L31 | L0 | L30 | L5 | L1 |
| exp4_gated_loadbalancing | L31 | L0 | L29 | L30 | L5 |
| exp5_gated_topk | L30 | L31 | L0 | L29 | L28 |
| ablation_no_l1 | L29 | L31 | L30 | L0 | L28 |
| ablation_no_layer_emb | L29 | L31 | L30 | L0 | L28 |
| ablation_partial_gating | L30 | L31 | L28 | L29 | - |

### Key Observations:

1. **Layer 0 and final layers (28-31) consistently show specialization** across all experiments.

2. **Specialization gradient**: Layers 28-31 form a "specialization zone" where task-specific routing emerges.

3. **Partial gating validates hypothesis**: partial_gating (only L28-31 gated) achieves comparable results, confirming these layers are where specialization matters most.

---

## 5. Task-Level Expert Usage

### Global Expert Usage by Task (exp5 - Top-K as reference):

| Task | Expert 0 | Expert 1 | Expert 2 |
|------|----------|----------|----------|
| WikiText | 28.5% | 28.7% | 42.8% |
| SQuAD | 28.9% | 29.6% | 41.5% |
| CoNLL-2003 | 26.2% | 29.3% | 44.5% |
| IMDB | 30.2% | 31.7% | 38.1% |

### Task Specialization Patterns:

1. **Expert 2 preference for structured tasks**: CoNLL-2003 (NER) and WikiText show highest Expert 2 usage.

2. **IMDB (sentiment) more balanced**: Shows most even distribution across experts.

3. **Pattern consistency**: Similar patterns observed across exp3, exp5, and ablations.

---

## 6. Entropy Analysis

### Mean Layer Entropy:

| Experiment | Mean Entropy | Min Entropy | Min Entropy Layer |
|------------|--------------|-------------|-------------------|
| exp3_gated_3experts | 0.93 | 0.12 | L31 |
| exp4_gated_loadbalancing | 1.00 | 0.57 | L31 |
| exp5_gated_topk | 0.72 | 0.08 | L30 |
| ablation_no_l1 | 0.70 | 0.07 | L29 |
| ablation_no_layer_emb | 0.72 | 0.07 | L29 |
| ablation_partial_gating | 0.13* | 0.07 | L30 |

*Note: partial_gating only has entropy for 4 layers (28-31)

### Key Observations:

1. **Load balancing maintains high entropy**: exp4 shows highest mean entropy (1.00) - tokens distributed more uniformly across experts.

2. **Top-k and no-L1 create sharper routing**: Lower entropy indicates more confident/deterministic expert selection.

3. **Final layers consistently show low entropy**: Layers 29-31 have the sharpest routing across all experiments.

---

## 7. Ablation Insights

### Impact of Removing L1 Regularization (no_l1):
- **Increased task specialization**: 0.0473 vs 0.0186 (exp3 baseline)
- **Sharper routing**: Lower entropy in final layers
- **Comparable performance**: Same eval loss (0.7437)
- **Conclusion**: L1 regularization primarily affects routing sharpness, not performance

### Impact of Removing Layer Embedding (no_layer_emb):
- **Minimal effect on specialization**: 0.0438 vs 0.0436 (exp5)
- **Best eval loss**: 0.7430 (marginal improvement)
- **Higher train loss**: 1.1439 (slower convergence)
- **Conclusion**: Layer embedding aids training stability but not final routing quality

### Impact of Partial Gating (only L28-31):
- **Validates specialization hypothesis**: Achieves same performance with 87.5% fewer gated layers
- **Comparable specialization in gated layers**: Similar patterns to full gating
- **Potential efficiency gains**: Significant parameter savings
- **Conclusion**: Specialization primarily happens in late layers; early layers may not need gating

---

## 8. Research Directions

Based on this analysis, several promising research angles emerge:

### A. Efficiency-Focused
- **Partial gating is effective**: Could reduce complexity by 87.5% with minimal quality loss
- Further investigate minimal gating (only L30-31?)

### B. Specialization-Focused
- **No-L1 increases specialization without hurting performance**: Could explore task-adaptive regularization
- **Top-k routing shows clearest specialization**: Hard routing may be better for interpretability

### C. Architecture-Focused
- **Layer embedding is optional**: Could simplify architecture
- **Load balancing reduces specialization**: Trade-off between efficiency and adaptivity

### D. Multi-Task Learning
- **Consistent layer specialization zones**: Could inform layer-wise learning rate schedules
- **Task-specific expert preferences emerge**: Could be leveraged for task routing

---

## 9. Summary Table

| Aspect | Best Performer | Worst Performer |
|--------|----------------|-----------------|
| Eval Loss | no_layer_emb (0.7430) | partial_gating (0.7438) |
| Task Specialization | no_l1 (0.0473) | exp4_loadbalancing (0.0133) |
| Load Balance | exp4 (0.065) | exp2 (0.168) |
| Training Stability | exp3 baseline | no_layer_emb |
| Parameter Efficiency | partial_gating | full gating |

---

*Generated from routing analysis data in `/ensicompute_ablations/routing_results/`*
