# Gated LoRA vs Baseline LoRA - Rapport de Comparaison

## Résumé Exécutif

| Métrique | Baseline LoRA | Gated LoRA | Différence |
|----------|---------------|------------|------------|
| **Eval Loss Finale** | 2.5750 | **2.5639** | -0.43% |
| **Perplexité Finale** | 13.13 | **12.98** | -1.14% |
| **Params Entraînables** | 5.24M (0.19%) | 6.56M (0.24%) | +25% |
| **Temps d'Entraînement** | 3h37min | 4h49min | +33% |
| **VRAM Utilisée** | 5.23 GB | 5.24 GB | ~0% |

**Conclusion**: Le Gated LoRA obtient une meilleure performance finale (-0.43% loss) mais au prix d'un temps d'entraînement plus long (+33%).

---

## 1. Configuration de l'Expérience

### Modèle de Base
- **Architecture**: Microsoft Phi-2 (2.78B paramètres)
- **Précision**: bfloat16
- **GPU**: NVIDIA A40 (46GB VRAM)
- **Dataset**: WikiText-2-raw-v1

### Hyperparamètres Communs
| Paramètre | Valeur |
|-----------|--------|
| Batch Size | 8 |
| Learning Rate | 1e-4 |
| Epochs | 3 |
| Max Length | 512 |
| Warmup Steps | 100 |
| Scheduler | Cosine |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Gradient Clipping | 0.5 |

### Configuration LoRA
| Paramètre | Baseline | Gated |
|-----------|----------|-------|
| Rank (r) | 16 | 16 (experts: 8, 16, 32) |
| Alpha | 32 | 32 |
| Dropout | 0.1 | 0.1 |
| Target Modules | q_proj, v_proj | q_proj, v_proj |
| Gating Hidden Dim | - | 256 |
| Load Balancing | - | 0.01 |

---

## 2. Évolution de la Loss

### Eval Loss par Step

| Step | Baseline | Gated | Delta |
|------|----------|-------|-------|
| 500 | 2.7767 | 2.7598 | -0.61% |
| 1000 | 2.7045 | 2.6905 | -0.52% |
| 1500 | 2.6740 | 2.6490 | -0.94% |
| 2000 | 2.6428 | 2.6235 | -0.73% |
| 2500 | 2.6063 | 2.5961 | -0.39% |
| 3000 | 2.5946 | 2.5878 | -0.26% |
| 3500 | 2.6080 | 2.5858 | -0.85% |
| 4000 | 2.5943 | 2.5815 | -0.49% |
| 4500 | 2.5820 | 2.5704 | -0.45% |
| 5000 | 2.5794 | 2.5685 | -0.42% |
| 5500 | 2.5756 | 2.5660 | -0.37% |
| 6000 | 2.5754 | 2.5645 | -0.42% |
| 6500 | 2.5755 | 2.5648 | -0.42% |
| 7000 | 2.5757 | 2.5647 | -0.43% |
| 7500 | 2.5750 | 2.5639 | -0.43% |
| 8000 | 2.5754 | 2.5644 | -0.43% |
| 8500 | 2.5756 | 2.5641 | -0.45% |
| **Final** | **2.5750** | **2.5639** | **-0.43%** |

### Observations

1. **Convergence plus rapide du Gated LoRA**: Dès le step 500, le Gated LoRA montre une avance.

2. **Plateau atteint vers step 5000-6000**: Les deux modèles convergent après ~60% de l'entraînement.

3. **Avantage constant**: Le Gated LoRA maintient un avantage de ~0.4-0.5% tout au long.

4. **Pas de divergence**: Aucun signe d'overfitting ou d'instabilité.

---

## 3. Analyse des Paramètres

### Baseline LoRA
```
Total Parameters:      2,784,926,720
Base Model (frozen):   2,779,683,840
LoRA Adapters:         5,242,880 (0.19%)
```

### Gated LoRA
```
Total Parameters:      2,785,583,107
Base Model (frozen):   2,779,683,840
LoRA Adapters:         5,242,880
Gating Network:        656,387 (0.66M)
-----------------------------------
Total Trainable:       6,555,654 (0.24%)
```

### Rapport Coût/Bénéfice
- **Surcoût en paramètres**: +1.31M (+25%)
- **Amélioration de loss**: -0.43%
- **Ratio**: ~3M paramètres supplémentaires pour 1% d'amélioration

---

## 4. Performance Temporelle

| Métrique | Baseline | Gated |
|----------|----------|-------|
| Temps Total | 3h 37min | 4h 49min |
| Temps/Step | ~1.17s | ~1.66s |
| Overhead | - | +42% |

### Analyse du Overhead

Le Gated LoRA est plus lent à cause de:
1. **Forward pass supplémentaire** pour le gating network
2. **Calcul des poids d'experts** à chaque token
3. **Load balancing loss** computation

---

## 5. Utilisation VRAM

| Métrique | Baseline | Gated |
|----------|----------|-------|
| Allocated | 5.23 GB | 5.24 GB |
| Reserved | 32.5 GB | 10.3 GB |

**Note**: La VRAM allouée est quasi-identique. La différence de "reserved" est due au caching PyTorch.

---

## 6. Métriques Spécifiques au Gated LoRA

### Load Balance Loss
```
Step 10:  0.00186
Step 50:  0.00003
Step 100: 0.00001
...
Final:    ~0.00002
```

**Observation**: Le load balance loss converge très rapidement vers 0, indiquant que le gating network apprend à équilibrer les experts. Cependant, cette valeur très basse suggère que le load balancing pourrait être **trop fort** (weight=0.01), forçant une distribution uniforme plutôt qu'adaptative.

---

## 7. Conclusions

### Points Positifs du Gated LoRA
1. **Meilleure performance finale** (-0.43% loss, -1.14% perplexité)
2. **Convergence légèrement plus rapide** dans les premiers steps
3. **Pas de surcoût VRAM significatif**
4. **Architecture plus expressive** avec capacité adaptative

### Points Négatifs du Gated LoRA
1. **Temps d'entraînement +33%** plus long
2. **Complexité accrue** (debugging, tuning)
3. **Avantage marginal** sur WikiText-2 (dataset homogène)

### Recommandations

1. **Réduire le load_balancing_weight** de 0.01 à 0.001
   - Permettrait au gating d'être plus sélectif

2. **Tester sur des datasets hétérogènes**
   - Code + texte naturel
   - Multi-domaine (médical, juridique, technique)
   - Le Gated LoRA devrait briller sur ces cas

3. **Analyser les routing patterns**
   - Visualiser quels experts sont activés pour quels tokens
   - Vérifier si le gating apprend quelque chose de significatif

4. **Optimiser le forward pass**
   - Le overhead de 42% est élevé
   - Considérer une implémentation plus efficace du gating

---

## 8. Prochaines Étapes Suggérées

### Court Terme
- [ ] Réduire load_balancing_weight et re-tester
- [ ] Ajouter logging des routing statistics par batch
- [ ] Créer script de visualisation des experts

### Moyen Terme
- [ ] Test sur dataset multi-domaine (ex: The Pile subset)
- [ ] Comparer avec d'autres valeurs de expert_ranks
- [ ] Implémenter top-k routing (sparse MoE)

### Long Terme
- [ ] Benchmark contre autres méthodes PEFT (AdaLoRA, QLoRA)
- [ ] Étudier le transfer learning du gating network
- [ ] Publication des résultats

---

## Annexe: Commandes de Reproduction

```bash
# Baseline
sbatch train.sh baseline

# Gated
sbatch train.sh gated

# Avec config personnalisée
python train.py --config configs/custom.json --wandb-mode offline
```

---

*Rapport généré le 5 décembre 2025*
*Expériences exécutées sur Ensimag ensicompute (NVIDIA A40)*
