# MERGE_NOTES.md — Plan de fusion des 11 dossiers `ensicompute_*`

> Document préparatoire au refactor vers `gated-lora-research-paper-001`.
> Synthèse des investigations menées le 2026-04-29.

---

## TL;DR

- **4 fichiers identiques** partout (~2400 LoC) : socle stable, rien à merger.
- **5 fichiers divergents** mais réconciliables :
  - 2 fichiers (`lora_experts.py`, `gated_lora_v2.py`) → divergences = **support architectures** (GQA, GPT-NeoX) + **flags d'ablation**.
  - 1 fichier (`gated_trainer.py`) → 2 extensions **orthogonales** (routing analysis + SLURM job chaining).
  - 1 fichier (`multi_task_dataset.py`) → superset (8 tâches contient les 4 originales).
  - 1 fichier (`gating_network.py`) → 1 ligne de diff, trivial.
- **Diff serveur ↔ local** : negligible. Seules diffs = `batch_size`/`grad_accum` dans `config.py` (tuning par modèle) et chemins SLURM dans `run_experiment.sh`. **Le local est plus récent.**

---

## 1. Code stable (identique sur les 11 dossiers)

| Fichier | LoC | Rôle |
|---------|-----|------|
| `src/models/base_model.py` | 352 | Wrapper HuggingFace, freeze base weights |
| `src/models/gated_lora.py` | 531 | Implémentation V1 (legacy) |
| `src/training/trainer.py` | 651 | Trainer LoRA standard (sans gating) |
| `src/analysis/routing_analysis.py` | 833 | Outils d'analyse routing post-training |

→ **Recopier directement** dans le nouveau repo, aucun choix à faire.

---

## 2. `gated_lora_v2.py` (6 variantes, 639–829L) — **Architecture + Ablations**

### Groupes sémantiques

| Groupe | Variantes | Particularité |
|--------|-----------|---------------|
| **A. Phi-2 baseline (767L)** | `harder_multitask`, `multirun` | `use_layer_embedding` + `gated_layers` + partial gating + `from_pretrained()` complet |
| **B. Ablations (743L)** | `ablations` | Phi-2 sans `use_layer_embedding` ni `gated_layers` |
| **C. Per-layer (639L)** | `per_layer`, `per_layer_multirun` | Phi-2 minimal + bug `mean_dominance` (typo, à corriger) |
| **D. GQA models (797L)** | `Qwen`, `SmolLM`, `Gemma-2`, `Llama-3.2`, `Llama3.2_modified` | + GQA dims, + cache_key simplifié, hooks layer-finding standard |
| **E. Pythia (829L)** | `Pythia-410M` | Tout du groupe D + branches GPT-NeoX (`gpt_neox.layers`, `query_key_value`, `dense_h_to_4h`/`dense_4h_to_h`) |

### Diffs concrètes (vs baseline 767L)

**Groupes B/C** : suppressions (features Phi-2 récentes pas encore intégrées dans ces forks expérimentaux).
**Groupes D/E** : ajouts de support architectural :

```python
# Ajouté dans D/E :
self.num_attention_heads = getattr(self.config, 'num_attention_heads', None)
self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_attention_heads)
self.head_dim = self.hidden_size // self.num_attention_heads
# + LoRAExpertPool reçoit ces dims
# + cache_key = layer_idx (au lieu de (layer_idx, tensor_ptr))
# + input_dim validation (Pythia)
# + branches gpt_neox.layers / query_key_value / dense_*_to_* (Pythia uniquement)
```

### Stratégie de merge

**Un seul fichier `gated_lora_v2.py`** combinant :
- Base = 767L (`harder_multitask`).
- Branchements architecturaux par detection automatique (`hasattr(model, 'gpt_neox')`, GQA params).
- Flags d'ablation via `__init__` params (`use_layer_embedding`, `gated_layers`).
- Cache key = `layer_idx` simple (la version GQA).
- Bug `mean_dominance` → corrigé (`mean_top1_dominance`).

**Estimé** : ~830 LoC final (~Pythia size).

---

## 3. `lora_experts.py` (5 variantes, 387–423L) — **Architecture-aware**

| Variante | LoC | Particularité |
|----------|-----|---------------|
| Phi-2 baseline | 387 | Hardcoded `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/dense/fc1/fc2` |
| Qwen/SmolLM/Llama | 415 | + GQA params (`num_attention_heads`, `num_key_value_heads`, `head_dim`), refactor en `_get_module_dimensions()` |
| Gemma-2 | 416 | Idem (1L de commentaire diff) |
| Pythia | 423 | + branches `query_key_value` (3*hidden), `dense_h_to_4h`, `dense_4h_to_h` |

### Diffs concrètes

**Centralisation des dims module** :
```python
# Avant (387L) : if-elif inline dans constructeur (27L)
# Après (415-423L) : LoRAExpert._get_module_dimensions(module_name) -> (in_dim, out_dim)
#   - q_proj : (hidden, num_heads * head_dim)
#   - k_proj/v_proj : (hidden, num_kv_heads * head_dim)  ← GQA!
#   - o_proj : (num_heads * head_dim, hidden)
#   - dense_h_to_4h / dense_4h_to_h (Pythia)
#   - query_key_value (Pythia, dim = 3*hidden)
```

### Stratégie de merge

**Un seul `LoRAExpert`** avec :
- Tous les params GQA optionnels (fallback `head_dim = hidden // num_heads`).
- `_get_module_dimensions()` couvrant **tous** les module_names rencontrés (Phi-2, GQA, GPT-NeoX).
- ~423 LoC final.

---

## 4. `gated_trainer.py` (3 variantes, 679–890L) — **2 extensions ORTHOGONALES**

### Variantes

| Variante | LoC | Spécialité |
|----------|-----|------------|
| Standard | 679 | 2-phase training (warmup + joint), routing stats logging |
| `per_layer_multirun` | 890 | + `RoutingSnapshot` + `run_routing_analysis()` (observabilité fine) |
| `Llama3.2_modified` | 858 | + SLURM job chaining (auto-resume + time limits) |

### Extension A : Routing analysis (per_layer_multirun, +211L)

**Ajouts** :
- `RoutingSnapshot` (dataclass) → capture état routing à un instant T.
- `set_analysis_dataloader()` → setter pour dataloader dédié.
- `run_routing_analysis()` (137L) → snapshot per-layer, per-task : entropie, specialization, gate_weights.
- `_save_routing_history()` → JSON serialization.
- Hook dans `train()` : appelle `run_routing_analysis()` tous les N steps.

### Extension B : SLURM chaining (Llama3.2_modified, +179L) — **CRITIQUE**

> **Cette extension est exactement ce dont le user a besoin pour gérer les time limits SLURM.**

**Ajouts** :
- `DEFAULT_MAX_RUNTIME_SECONDS = 3.5h` (safety margin pour partition 4h).
- `TrainingState.batch_idx` → resume mid-epoch précis.
- `find_latest_checkpoint()` → auto-discovery du dernier checkpoint.
- `train(resume_from_checkpoint="auto"|path)` → resume orchestré.
- Batch skipping logic → ne re-traite pas les batches déjà vus.
- Time limit guard → `break` propre avant timeout SLURM.
- `_mark_training_done()` + `is_training_done()` → marqueur file-based pour orchestration multi-job.
- Checkpoint JSON étendu (`batch_idx`, `total_train_loss`, `num_train_steps`).

### Stratégie de merge

Les deux extensions sont **orthogonales** (l'une = observabilité, l'autre = orchestration). On intègre **les deux** :

- **SLURM chaining → toujours actif** (avec `max_runtime_seconds` configurable).
- **Routing analysis → flag config** `enable_routing_analysis: bool = false` (overhead nul si désactivé).

**Résultat estimé** : ~1000 LoC final (679 + 211 + 179 - duplications).

**Ordre de merge** :
1. Apply Llama3.2_modified (SLURM foundation).
2. Port `RoutingSnapshot` + `run_routing_analysis()` (avec feature flag).
3. Tester `(SLURM_on, analysis_on)` + `(SLURM_on, analysis_off)`.

---

## 5. `multi_task_dataset.py` (2 variantes, 542–853L) — **Superset strict**

| Variante | LoC | Tâches |
|----------|-----|--------|
| 4-task baseline | 542 | SQuAD, IMDB, CoNLL-2003, WikiText-2 |
| 8-task harder | 853 | + GSM8K, XSum, CommonsenseQA, MNLI |

### Tâches ajoutées (8-task)

| Tâche | Type | Source HF | Difficulté |
|-------|------|-----------|------------|
| GSM8K | Math reasoning | `gsm8k/main` | 0.9 |
| XSum | Summarization | `xsum` | 0.7 |
| CommonsenseQA | Commonsense reasoning | `commonsense_qa` | 0.8 |
| MNLI | NLI | `glue/mnli` | 0.6 |

### Refactors structurels (uniquement dans 853L)

- `TASK_CATEGORIES` (classification par type)
- `TASK_COMPLEXITY` (scores 0.3–0.9)
- Presets : `get_original_4_tasks()`, `get_harder_4_tasks()`, `get_all_8_tasks()`, `get_diverse_6_tasks()`, `get_reasoning_focused()`

### Stratégie de merge

**Base = 853L**, exposer les configurations via YAML :

```yaml
# configs/tasks/original_4.yaml
datasets: [squad, imdb, conll2003, wikitext]
weights: [0.30, 0.25, 0.25, 0.20]

# configs/tasks/all_8.yaml
datasets: [squad, imdb, conll2003, wikitext, gsm8k, xsum, commonsenseqa, mnli]
weights: [0.12, 0.10, 0.12, 0.10, 0.15, 0.14, 0.14, 0.13]
```

**Estimé** : ~30 min de travail.

---

## 6. `gating_network.py` (2 variantes, 539–540L) — **Trivial**

1 ligne de diff entre groupes (B/C) et le reste. À investiguer rapidement au moment du merge — probablement une refacto cosmétique.

---

## 7. `config.py` (5 variantes par modèle) — **Hyperparams**

Toutes les diffs serveur↔local et entre dossiers concernent **uniquement** :
- `batch_size`
- `gradient_accumulation_steps`
- Adaptés par modèle (Gemma-2 : 4/8 ; Llama-3.2 : 8/4 ; Pythia : 6/5 ; SmolLM : 6/5 ; Phi-2 : 8/4).

→ **À transformer en YAML** : `configs/models/<model>.yaml` avec ces valeurs.

---

## 8. Architecture cible du nouveau repo

```
gated-lora-research-paper-001/
├── README.md                       # Vue d'ensemble + repro instructions
├── pyproject.toml                  # uv-managed deps
├── .gitignore                      # outputs/, wandb/, *.pt, __pycache__/, .venv/
├── src/gated_lora/
│   ├── __init__.py
│   ├── models/
│   │   ├── base_model.py           # ← stable
│   │   ├── gated_lora.py           # ← stable (V1 legacy, à conserver pour repro)
│   │   ├── gated_lora_v2.py        # ← merge des 6 variantes (~830L)
│   │   ├── gating_network.py       # ← stable (1L diff à clarifier)
│   │   └── lora_experts.py         # ← merge des 5 variantes (~423L)
│   ├── data/
│   │   └── multi_task_dataset.py   # ← merge des 2 variantes (base = 853L)
│   ├── training/
│   │   ├── config.py               # ← stub minimal, lit YAML
│   │   ├── trainer.py              # ← stable
│   │   └── gated_trainer.py        # ← merge des 3 variantes (~1000L)
│   ├── analysis/
│   │   ├── routing_analysis.py     # ← stable
│   │   └── routing_snapshot.py     # ← extracted from per_layer_multirun
│   └── utils/
│       └── logging.py              # ← stable
├── configs/
│   ├── models/
│   │   ├── phi2.yaml
│   │   ├── gemma2.yaml
│   │   ├── llama32.yaml
│   │   ├── llama32_modified.yaml
│   │   ├── pythia410m.yaml
│   │   ├── qwen25_05b.yaml
│   │   └── smollm360m.yaml
│   ├── tasks/
│   │   ├── original_4.yaml
│   │   ├── harder_4.yaml
│   │   ├── all_8.yaml
│   │   └── reasoning_focused.yaml
│   ├── ablations/
│   │   ├── no_layer_embedding.yaml
│   │   ├── partial_gating.yaml
│   │   └── per_layer_only.yaml
│   └── experiments/
│       ├── phi2_harder_multitask.yaml      # combinaison
│       ├── gemma2_all_8.yaml
│       └── ...
├── scripts/
│   ├── slurm/
│   │   ├── train.sbatch            # template paramétré (--config, --partition)
│   │   ├── analysis.sbatch
│   │   └── chain_jobs.sh           # orchestration multi-job (utilise SLURM chaining feature)
│   └── launch_experiment.sh        # wrapper
├── experiments/                    # tracking : juste .json/.md résultats (PAS les checkpoints)
│   └── README.md
├── paper/
│   ├── figures/
│   └── tex/
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
└── train.py                        # entrypoint principal: `uv run python train.py --config configs/experiments/X.yaml`
```

---

## 9. Plan d'exécution proposé (phases successives, validation entre chaque)

### Phase A — Setup repo (30 min)
- `gh repo create gated-lora-research-paper-001 --private --clone`
- `pyproject.toml` avec `uv` + deps (transformers, torch, datasets, accelerate, peft, wandb).
- `.gitignore` strict (no checkpoints, no wandb, no logs).
- Squelette de dossiers.
- 1er commit "chore: scaffold".

### Phase B — Code stable (45 min)
- Recopier les 4 fichiers identiques partout (`base_model.py`, `gated_lora.py`, `trainer.py`, `routing_analysis.py`).
- Recopier `utils/logging.py`.
- 1 commit "feat: import stable code from ensicompute_harder_multitask".

### Phase C — Merge `lora_experts.py` (1h)
- Base = Pythia (423L, le plus complet).
- Vérifier que les branches Phi-2 baseline restent fonctionnelles (`q_proj`/`fc1`/`fc2`/`dense`).
- Test : instancier `LoRAExpert` pour Phi-2, Qwen, Llama, Gemma, Pythia → toutes dims correctes.
- 1 commit "feat: unified LoRAExpert with multi-architecture support".

### Phase D — Merge `gated_lora_v2.py` (2h)
- Base = `harder_multitask` (767L, Phi-2 complet).
- Intégrer GQA support (depuis Qwen).
- Intégrer Pythia branches (`gpt_neox.layers`, `query_key_value`, `dense_h_to_4h`).
- Cache key = `layer_idx` (version GQA).
- Corriger bug `mean_dominance` → `mean_top1_dominance`.
- Tests : forward pass sur Phi-2, Qwen, Pythia.
- 1 commit "feat: unified GatedLoRAModelV2 supporting Phi-2/GQA/GPT-NeoX".

### Phase E — Merge `gated_trainer.py` (2h)
- Base = `harder_multitask` (679L).
- Intégrer SLURM chaining (depuis Llama3.2_modified) → **toujours actif**.
- Intégrer routing analysis (depuis per_layer_multirun) → **flag config**.
- Tests : training run minimal (1 step) + resume from checkpoint + time limit trigger.
- 1 commit "feat: unified gated trainer with SLURM chaining + optional routing analysis".

### Phase F — Merge `multi_task_dataset.py` (45 min)
- Base = `harder_multitask` (853L).
- Tâches paramétrées via YAML (configs/tasks/).
- 1 commit "feat: unified dataset with YAML-driven task configuration".

### Phase G — Configs YAML (1h30)
- Modèles (7 fichiers).
- Tâches (4 fichiers).
- Ablations (3 fichiers).
- Expériences (combinaisons utilisées dans les anciennes expés).
- 1 commit "feat: YAML configs for models/tasks/ablations".

### Phase H — Scripts SLURM (1h)
- `train.sbatch` paramétré (`--config`, `--partition`, `--time`, `--seed`).
- `chain_jobs.sh` : orchestration multi-job utilisant `find_latest_checkpoint` + `is_training_done()`.
- README scripts/.
- 1 commit "feat: parameterized SLURM templates with auto-chaining".

### Phase I — Tests & docs (1h30)
- 3 tests : modèle (forward), data (loading), training (1 step).
- README principal (run instructions, paper info).
- `experiments/README.md` (où sont les résultats — pointe vers le disque externe).
- 1 commit "test: add smoke tests + docs".

### Phase J — Push & validation finale (30 min)
- `git push -u origin main`
- Vérifier que tout est privé.
- Tag `v0.1.0-pre-paper`.

**Total estimé : ~12h de travail concentré.**

---

## 10. Risques / points d'attention

1. **Bug Pythia 829L → 797L** : Pythia a 32L de plus que Qwen/Gemma. Vérifier au merge que ces 32L sont bien Pythia-spécifiques (gpt_neox + query_key_value) et pas un truc général qu'on perdrait en branchant sur "GQA only".
2. **Llama3.2_modified vs Llama3.2** : la diff principale est dans `gated_trainer.py`. Mais quelle expérience donnait le "modified" ? À comprendre — peut-être que c'est juste le déploiement SLURM-chained du même modèle.
3. **Tests reproducibilité** : avant de supprimer le code des dossiers ensicompute_*, faire tourner UNE expé déjà connue avec le nouveau repo pour valider qu'on obtient les mêmes résultats (même seed, même config).
4. **Disque externe = SPOF** : avant de toucher au serveur, faire un backup du backup (autre disque ou cloud) si possible. Sinon, accepter le risque.

---

## Annexe — Fichiers locaux uniquement (à intégrer aussi)

| Fichier | Origine | Destination dans nouveau repo |
|---------|---------|-------------------------------|
| `ensicompute/AUDIT_REPORT.md` | Local | `experiments/legacy/audit_report.md` |
| `ensicompute/COMPARISON_REPORT.md` | Local | `experiments/legacy/comparison_report.md` |
| `ensicompute/IMPROVEMENTS.md` | Local | `experiments/legacy/improvements.md` |
| `ensicompute/README.md` | Local | À fusionner dans README principal |
| `ensicompute_per_layer/FUTURE_WORK.md` | Local | `paper/future_work.md` |
| `ensicompute_per_layer/README.md` | Local | `experiments/legacy/per_layer_readme.md` |
| `ensicompute_multirun/SESSION_REPORT.md` | Local | `experiments/legacy/multirun_session.md` |
| `ensicompute_ablations/routing_results/*` | Local | `experiments/legacy/ablations_routing/` |
| Tous les `analysis_results/*.md`, `figures/*` | Local | `experiments/legacy/<expe>/` |
