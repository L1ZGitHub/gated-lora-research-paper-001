# Gated LoRA: per-layer expert routing for parameter-efficient multi-task adaptation

**Author**: Helain Zimmermann (M2 Ensimag, Grenoble INP)
**Contact**: helain.zimmermann@grenoble-inp.org
**Code**: github.com/L1ZGitHub/gated-lora-research-paper-001 (public, MIT)
**Checkpoints**: huggingface.co/datasets/Helain/gated-lora-experiments
**Status**: pipeline + 36/36 smoke tests green; experiments queued.

## Problem

LoRA (Hu et al., 2022) fine-tunes a transformer by injecting low-rank adapters
$\Delta W = BA$ (rank $r$) into a small set of projections. In *multi-task*
fine-tuning the practitioner picks a single rank $r$ and the same adapter is
re-used for tasks as diverse as code generation, math, summarization, and
sentence classification. The mismatch between heterogeneous task demands and a
single fixed-rank adapter is largely unaddressed in the PEFT literature.

## Proposal

**Gated LoRA** replaces each LoRA injection point with a **pool of $K$ LoRA
adapters of different ranks** (e.g. $r \in \{8, 16, 32\}$), gated by a small
per-layer MLP $g_\ell(h)$ that, given the hidden state $h$, produces a soft
mixture $\alpha_\ell \in \Delta^{K-1}$ over the $K$ experts. The effective
update at layer $\ell$, module $m$ is

$$\Delta W_{\ell,m} = \sum_{k=1}^{K} \alpha_{\ell,k}(h)\, B_{\ell,m,k} A_{\ell,m,k}.$$

The gate sees a *layer-index embedding* concatenated with $h$, enabling
position-conditioned routing. An L1 penalty on $\alpha$ encourages sparse
allocation, so most tokens are processed by the cheap $r=8$ expert and only
"hard" tokens recruit the high-rank experts.

## Hypotheses

1. **Capacity-allocation hypothesis** — for a fixed total LoRA-parameter
   budget, Gated LoRA matches or beats vanilla LoRA on a heterogeneous
   multi-task mix because expert capacity is routed where it helps.
2. **Layer-task specialization** — the learned routing $\alpha_\ell$ exhibits
   stable, reproducible-across-seeds patterns that align with known
   transformer-layer roles (early layers ↔ tokenization-style; mid ↔
   syntactic; late ↔ semantic / task-conditioned).
3. **Hard-task / hard-layer correlation** — harder tasks (GSM8K math,
   CommonsenseQA) recruit the high-rank expert more than easier tasks
   (IMDB sentiment, WikiText LM), and the recruitment concentrates in
   specific layers rather than spreading uniformly.

## Experimental matrix

- **7 base models, 350M–3B params**: Phi-2, Gemma-2-2B, Llama-3.2-3B,
  Pythia-410M, Qwen2.5-0.5B, SmolLM-360M (+ optional Llama-3.2-3B variant
  with custom long-runtime trainer for cross-checking).
- **8-task mixture** (HuggingFace standard): SQuAD, IMDB, CoNLL-2003,
  WikiText-2, GSM8K, XSum, CommonsenseQA, MNLI. Two task subsets compared
  (4-task baseline, 8-task harder).
- **3 ablations** isolating design choices: (i) no layer-index embedding,
  (ii) no L1 sparsity, (iii) partial gating (gate only early + late
  layers).
- **3 seeds per (model × task-config × ablation)** for paper-grade
  reproducibility.

Total: ≈ 60 training runs of 8–12 h each.

## Compute request

≈ **600 GPU-h on H100 80 GB normalisées** (Jean Zay Accès Dynamique),
distributed across the 60 runs above. Each run streams checkpoints to a
private Hugging Face dataset so the experiments survive any interruption.
Storage on Jean Zay: < 50 GB peak (transient — everything lives long-term
on HF Hub).

## Deliverables

- **Public code** (already on GitHub, MIT).
- **Public checkpoints + per-step routing snapshots** on the HF dataset on
  paper submission.
- **Preprint** targeting an ML workshop or conference (timeline: Q3 2026).
- **Acknowledgement** of GENCI / IDRIS in line with eDARI policy.

## Why this matters

PEFT methods are now the dominant adaptation strategy for foundation models
across academia and industry. A method that **allocates rank capacity
adaptively at no inference-time cost** would shift the LoRA Pareto frontier
(trainable params vs. multi-task accuracy). Independently of the empirical
gain, the routing patterns themselves are *interpretable* evidence for the
"which layers do which work" question that remains open in mechanistic
interpretability — making this work doubly relevant.
