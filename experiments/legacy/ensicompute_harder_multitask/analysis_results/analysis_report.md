# Comprehensive Checkpoint Analysis Report

Generated: 2025-12-20T12:18:05.977550

**Tasks analyzed:** squad, imdb, conll2003, wikitext, gsm8k, xsum, commonsenseqa, mnli

## Executive Summary

- **Highest task specialization:** exp5_gated_topk_seed1 (score: 0.0657)
- **Lowest task specialization:** exp4_gated_loadbalancing_seed1 (score: 0.0065)
- **Specialization range:** 0.0065 - 0.0657

## Per-Experiment Analysis

### ablation_no_l1_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
- Task specialization: 0.0103 → 0.0177
- Load imbalance: 0.0703 → 0.1121
- Max dominance: 0.8631 → 0.9662

**Final Checkpoint (Step 5000):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| squad | 24.27% | 31.49% | 44.25% | E2 |
| conll2003 | 24.89% | 30.53% | 44.58% | E2 |
| mnli | 24.20% | 30.63% | 45.18% | E2 |
| imdb | 24.35% | 33.29% | 42.36% | E2 |
| wikitext | 24.57% | 31.39% | 44.04% | E2 |
| gsm8k | 24.14% | 31.65% | 44.20% | E2 |
| xsum | 25.56% | 33.94% | 40.50% | E2 |
| commonsenseqa | 23.70% | 30.73% | 45.57% | E2 |

Top specializing layers: [31, 0, 28, 3, 1]
Max specialization: Layer 31 (score: 0.2622)

---

### exp4_gated_loadbalancing_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
- Task specialization: 0.0082 → 0.0065
- Load imbalance: 0.0109 → 0.0055
- Max dominance: 0.7024 → 0.4073

**Final Checkpoint (Step 5000):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| imdb | 33.52% | 33.56% | 32.92% | E1 |
| commonsenseqa | 33.00% | 32.68% | 34.32% | E2 |
| wikitext | 32.85% | 32.84% | 34.31% | E2 |
| conll2003 | 33.53% | 32.70% | 33.77% | E2 |
| squad | 32.72% | 33.02% | 34.25% | E2 |
| mnli | 33.49% | 32.81% | 33.70% | E2 |
| gsm8k | 33.03% | 33.03% | 33.94% | E2 |
| xsum | 32.84% | 33.75% | 33.41% | E1 |

Top specializing layers: [31, 30, 28, 6, 1]
Max specialization: Layer 31 (score: 0.1483)

---

### ablation_no_layer_emb_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
- Task specialization: 0.0133 → 0.0203
- Load imbalance: 0.0666 → 0.0842
- Max dominance: 0.8870 → 0.9700

**Final Checkpoint (Step 5000):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| gsm8k | 28.21% | 28.95% | 42.84% | E2 |
| mnli | 27.26% | 28.01% | 44.73% | E2 |
| imdb | 28.51% | 30.06% | 41.43% | E2 |
| conll2003 | 27.70% | 27.89% | 44.41% | E2 |
| commonsenseqa | 27.00% | 27.75% | 45.25% | E2 |
| squad | 27.41% | 28.82% | 43.78% | E2 |
| wikitext | 27.77% | 29.18% | 43.06% | E2 |
| xsum | 29.60% | 31.18% | 39.22% | E2 |

Top specializing layers: [31, 28, 0, 9, 1]
Max specialization: Layer 31 (score: 0.2451)

---

### exp3_gated_3experts_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
- Task specialization: 0.0142 → 0.0287
- Load imbalance: 0.0562 → 0.0764
- Max dominance: 0.8095 → 0.9073

**Final Checkpoint (Step 5000):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| mnli | 30.29% | 28.42% | 41.29% | E2 |
| gsm8k | 31.42% | 30.24% | 38.34% | E2 |
| wikitext | 31.96% | 30.24% | 37.80% | E2 |
| conll2003 | 30.05% | 28.57% | 41.38% | E2 |
| commonsenseqa | 30.03% | 28.46% | 41.51% | E2 |
| squad | 31.25% | 29.43% | 39.31% | E2 |
| xsum | 33.97% | 32.34% | 33.69% | E0 |
| imdb | 32.70% | 31.16% | 36.14% | E2 |

Top specializing layers: [31, 28, 0, 30, 29]
Max specialization: Layer 31 (score: 0.2535)

---

### exp2_gated_2experts_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
- Task specialization: 0.0160 → 0.0323
- Load imbalance: 0.1088 → 0.1258
- Max dominance: 0.9521 → 0.9975

**Final Checkpoint (Step 5000):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| mnli | 40.03% | 59.97% | 0.00% | E1 |
| conll2003 | 39.33% | 60.67% | 0.00% | E1 |
| commonsenseqa | 39.77% | 60.23% | 0.00% | E1 |
| squad | 42.72% | 57.28% | 0.00% | E1 |
| xsum | 46.87% | 53.14% | 0.00% | E1 |
| imdb | 45.04% | 54.96% | 0.00% | E1 |
| gsm8k | 43.53% | 56.47% | 0.00% | E1 |
| wikitext | 42.01% | 58.00% | 0.00% | E1 |

Top specializing layers: [28, 31, 29, 0, 1]
Max specialization: Layer 28 (score: 0.2651)

---

### exp5_gated_topk_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
- Task specialization: 0.0390 → 0.0657
- Load imbalance: 0.1256 → 0.0867
- Max dominance: 0.6899 → 0.4945

**Final Checkpoint (Step 4500):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| xsum | 38.95% | 34.85% | 26.20% | E0 |
| gsm8k | 35.14% | 38.81% | 26.05% | E1 |
| wikitext | 31.79% | 41.06% | 27.14% | E1 |
| commonsenseqa | 24.45% | 38.91% | 36.64% | E1 |
| imdb | 37.13% | 40.10% | 22.76% | E1 |
| conll2003 | 27.53% | 42.13% | 30.34% | E1 |
| mnli | 26.88% | 39.83% | 33.27% | E1 |
| squad | 35.47% | 39.73% | 24.80% | E1 |

Top specializing layers: [30, 24, 6, 28, 27]
Max specialization: Layer 30 (score: 0.3533)

---

### ablation_partial_gating_seed1

**Evolution:**
- Steps analyzed: [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
- Task specialization: 0.0062 → 0.0117
- Load imbalance: 0.0487 → 0.0511
- Max dominance: 0.9583 → 0.9388

**Final Checkpoint (Step 5000):**

Task-Expert Usage:
| Task | Expert 0 | Expert 1 | Expert 2 | Dominant |
|------|----------|----------|----------|----------|
| wikitext | 29.46% | 32.18% | 38.36% | E2 |
| imdb | 29.71% | 33.67% | 36.61% | E2 |
| conll2003 | 29.37% | 32.12% | 38.52% | E2 |
| gsm8k | 29.44% | 33.00% | 37.56% | E2 |
| mnli | 29.34% | 32.21% | 38.45% | E2 |
| commonsenseqa | 29.33% | 32.31% | 38.36% | E2 |
| xsum | 30.22% | 34.61% | 35.17% | E2 |
| squad | 29.36% | 32.94% | 37.70% | E2 |

Top specializing layers: [31, 30, 28, 3, 25]
Max specialization: Layer 31 (score: 0.2605)

---

## Key Findings

### Task Specialization Patterns

_Analysis of which experts specialize for which tasks at the final checkpoint._

**squad:**
- Expert 1: 2 experiments (exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 5 experiments (ablation_no_l1, exp4_gated_loadbalancing, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

**conll2003:**
- Expert 1: 2 experiments (exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 5 experiments (ablation_no_l1, exp4_gated_loadbalancing, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

**mnli:**
- Expert 1: 2 experiments (exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 5 experiments (ablation_no_l1, exp4_gated_loadbalancing, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

**imdb:**
- Expert 1: 3 experiments (exp4_gated_loadbalancing, exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 4 experiments (ablation_no_l1, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

**wikitext:**
- Expert 1: 2 experiments (exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 5 experiments (ablation_no_l1, exp4_gated_loadbalancing, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

**gsm8k:**
- Expert 1: 2 experiments (exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 5 experiments (ablation_no_l1, exp4_gated_loadbalancing, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

**xsum:**
- Expert 0: 2 experiments (exp3_gated_3experts, exp5_gated_topk)
- Expert 1: 2 experiments (exp4_gated_loadbalancing, exp2_gated_2experts)
- Expert 2: 3 experiments (ablation_no_l1, ablation_no_layer_emb, ablation_partial_gating)

**commonsenseqa:**
- Expert 1: 2 experiments (exp2_gated_2experts, exp5_gated_topk)
- Expert 2: 5 experiments (ablation_no_l1, exp4_gated_loadbalancing, ablation_no_layer_emb, exp3_gated_3experts, ablation_partial_gating)

## Recommendations

1. Experiments with highest task specialization show more distinct routing patterns
2. Load balancing reduces specialization but improves expert utilization
3. Top-K routing tends to produce sharper specialization
4. Specialization emerges primarily in final layers (28-31)
