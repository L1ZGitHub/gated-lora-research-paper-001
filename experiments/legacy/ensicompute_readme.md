# Gated LoRA - Ensicompute

Dossier autonome pour entraîner les modèles sur la plateforme GPU de l'Ensimag.

## Quick Start

```bash
# 1. Copier ce dossier sur nash (via scp, rsync, ou drag&drop)
scp -r ensicompute/ login@nash.ensimag.fr:~/gated-lora/

# 2. Se connecter à nash
ssh login@nash.ensimag.fr

# 3. Setup (une seule fois)
cd ~/gated-lora
bash setup.sh

# 4. Lancer l'entraînement
sbatch train.sh baseline   # ou "gated"

# 5. Surveiller
squeue -u $USER
tail -f logs/slurm_<JOB_ID>.out
```

## Structure

```
ensicompute/
├── setup.sh           # Setup environnement (run once)
├── train.sh           # Script SLURM principal
├── train.py           # Code d'entraînement
├── requirements.txt   # Dépendances Python
├── configs/
│   ├── baseline_full.json
│   └── gated_full.json
├── src/               # Code source
├── logs/              # Logs SLURM (créé automatiquement)
└── outputs/           # Checkpoints (créé automatiquement)
```

## Commandes utiles

```bash
# Voir les jobs en cours
squeue -u $USER

# Voir tous les jobs sur la plateforme
squeue

# Annuler un job
scancel <JOB_ID>

# Annuler tous ses jobs
scancel -u $USER

# Voir les ressources disponibles
sinfo

# Test interactif (sans sbatch)
srun --gres=shard:1 --cpus-per-task=4 --mem=16GB python -c "import torch; print(torch.cuda.is_available())"
```

## Ressources demandées

- **GPU**: 1 partition (`--gres=shard:1`)
- **CPU**: 4 cores
- **RAM**: 32 GB
- **Temps max**: 8 heures

## GPUs disponibles

| Noeud | GPU | VRAM | Dispo |
|-------|-----|------|-------|
| tesla | V100 | 32 GB | 1 |
| turing-1..11 | RTX 6000 | 24 GB | 33 |
| ampere | A40 | **46 GB** | 3 |

Pour cibler un GPU spécifique, édite `train.sh` et décommente la ligne `--nodelist` souhaitée.

## Troubleshooting

### CUDA not available
Vérifie que tu as bien `--gres=shard:1` dans ta commande srun/sbatch.

### Out of memory (OOM)
- Réduire `batch_size` dans la config (déjà à 1)
- Activer `load_in_4bit: true` dans la config
- Réduire `max_length` (512 → 256)

### Job pending longtemps
Normal si la plateforme est chargée. Utilise `squeue` pour voir la file d'attente.

### Module not found
Assure-toi d'avoir lancé `bash setup.sh` et que le venv est créé.
