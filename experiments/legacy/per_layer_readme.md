# Gated LoRA - Per-Layer Gating Experiments

**Version figée** : 2024-12-06
**Status** : Experiments 1-5 completed

Ce dossier contient le code figé pour les expériences de Gated LoRA avec gating per-layer.

## Structure

```
ensicompute_per_layer/
├── src/
│   ├── models/
│   │   ├── gated_lora_v2.py      # Modèle principal avec hooks per-module
│   │   ├── gating_network.py     # Réseau de gating avec layer embedding
│   │   └── lora_experts.py       # Pool d'experts LoRA
│   ├── training/
│   │   ├── config.py             # Configuration (inclut L1 regularization)
│   │   └── gated_trainer.py      # Trainer custom
│   ├── data/
│   │   └── multi_task_dataset.py # Dataset multi-tâches
│   └── analysis/
│       └── routing_analysis.py   # Analyse du routing per-layer, per-task
├── configs/
│   └── exp*.json                 # Configurations des 5 expériences
├── train_v2.py                   # Script d'entraînement principal
├── experiment_v2.sh              # Script SLURM pour lancer les expériences
├── analyze_routing_standalone.py # Analyse post-training
├── run_routing_analysis.sh       # Script SLURM pour analyse
├── requirements.txt              # Dépendances Python
├── FUTURE_WORK.md               # Pistes d'amélioration et angles de publication
└── README.md                     # Ce fichier
```

## Expériences

| ID | Nom | Description | Experts |
|----|-----|-------------|---------|
| 1 | Baseline | LoRA standard r=16 | - |
| 2 | Gated 2 experts | Gated LoRA | r=8, r=16 |
| 3 | Gated 3 experts | Gated LoRA | r=8, r=16, r=32 |
| 4 | + Load Balancing | Avec loss de balancing | r=8, r=16, r=32 |
| 5 | + Top-K | Routing sparse (k=2) | r=8, r=16, r=32 |

## Résultats Clés

### Performance
- **Exp3** (3 experts) : Eval loss **0.7435** (baseline: 0.7441)
- Les modèles gated égalent ou battent légèrement la baseline

### Spécialisation
- **Couches 30-31** montrent une spécialisation par tâche (score ~0.27)
- **IMDB** (sentiment) utilise des experts plus petits que les autres tâches
- **Couches 0-29** : routing quasi-uniforme

## Réplication

### Installation

```bash
cd ensicompute_per_layer
pip install -r requirements.txt
```

### Entraînement

```bash
# Sur cluster SLURM
sbatch experiment_v2.sh 1  # Baseline
sbatch experiment_v2.sh 2  # Gated 2 experts
sbatch experiment_v2.sh 3  # Gated 3 experts
sbatch experiment_v2.sh 4  # + Load Balancing
sbatch experiment_v2.sh 5  # + Top-K

# Ou directement
python train_v2.py --experiment 3 --use-multi-task --wandb-mode offline
```

### Analyse du Routing

```bash
# Sur cluster SLURM
sbatch run_routing_analysis.sh

# Ou directement
python analyze_routing_standalone.py --all --num-batches 100
```

## Modifications Clés par rapport à l'Implémentation Initiale

1. **Fix Bug Modules** : Chaque Linear (q_proj, k_proj, v_proj, dense) reçoit maintenant son propre LoRA delta (avant seul q_proj était modifié)

2. **Layer Embedding** : Le gating network reçoit un embedding de la couche pour différencier son comportement par profondeur

3. **L1 Regularization** : Régularisation L1 sur les gate weights pour encourager la sparsité

## Configuration

Paramètres clés dans `src/training/config.py` :

```python
# Gating
per_layer_gating: bool = True
gating_hidden_dim: int = 256
use_layer_embedding: bool = True  # NOUVEAU

# Regularization
use_load_balancing: bool = True
load_balancing_weight: float = 0.001
use_l1_gate_regularization: bool = True  # NOUVEAU
l1_gate_weight: float = 0.01  # NOUVEAU

# Routing
use_top_k: bool = False
top_k: int = 2
```

## Citation

*À compléter après publication*

## Voir Aussi

- `FUTURE_WORK.md` : Pistes d'amélioration et angles de publication
