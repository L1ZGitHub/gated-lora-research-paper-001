# Améliorations Proposées pour Gated LoRA

## Problème Principal

Sur WikiText-2, le Gated LoRA n'a qu'un avantage marginal (-0.43%) car:
1. **Dataset homogène** - WikiText-2 est principalement du texte encyclopédique
2. **Load balancing trop fort** - Force une distribution uniforme des experts
3. **Pas de différenciation des experts** - Tous utilisent le même alpha/dropout

---

## Amélioration 1: Réduire le Load Balancing

### Problème
Le `load_balancing_weight=0.01` force le gating à distribuer uniformément, empêchant la spécialisation.

### Solution
```json
{
  "model": {
    "load_balancing_weight": 0.001
  }
}
```

### Fichier à créer: `configs/gated_low_lb.json`

---

## Amélioration 2: Dataset Multi-Domaine

### Problème
WikiText-2 est trop homogène pour montrer l'intérêt du routing adaptatif.

### Solution
Utiliser un mix de datasets:
- **Code**: CodeSearchNet ou The Stack
- **Texte**: WikiText-2
- **Instructions**: Dolly/Alpaca

### Exemple de config
```json
{
  "data": {
    "dataset_name": "EleutherAI/pile",
    "dataset_config_name": "pile-10k",
    "text_column": "text"
  }
}
```

---

## Amélioration 3: Différencier les Experts

### Problème Actuel
Tous les experts ont le même `lora_alpha=32`, donc même "force" d'adaptation.

### Solution
Adapter l'alpha proportionnellement au rank:
- Expert 1 (r=8): alpha=16
- Expert 2 (r=16): alpha=32
- Expert 3 (r=32): alpha=64

Cela créerait une vraie hiérarchie de capacité.

### Modification dans `gated_lora.py`

```python
# Ligne ~257 dans _create_experts()
expert_alphas = [r * 2 for r in self.expert_ranks]  # [16, 32, 64]
```

---

## Amélioration 4: Top-K Routing (Sparse MoE)

### Problème
Le routing dense active tous les experts, diluant la spécialisation.

### Solution
Activer `use_top_k=True` avec `top_k=2` pour n'utiliser que 2 experts par token.

### Dans le forward pass
```python
outputs = self.model(
    input_ids=batch["input_ids"],
    attention_mask=batch.get("attention_mask"),
    labels=batch.get("labels", batch["input_ids"]),
    use_top_k=True,
    top_k=2,
)
```

---

## Amélioration 5: Logging des Routing Statistics

### Problème
On ne sait pas comment le gating route les tokens.

### Solution
Logger les statistiques de routing par batch:

```python
# Dans trainer.py, après train_step()
if "expert_weights" in outputs:
    expert_usage = outputs["expert_weights"].mean(dim=[0,1])
    metrics["expert_0_usage"] = expert_usage[0].item()
    metrics["expert_1_usage"] = expert_usage[1].item()
    metrics["expert_2_usage"] = expert_usage[2].item()
```

---

## Amélioration 6: Visualisation des Experts

### Script de visualisation

```python
# visualize_routing.py
import torch
import matplotlib.pyplot as plt

def visualize_expert_routing(model, tokenizer, text):
    """Visualise quels experts sont activés pour chaque token."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, return_routing_info=True)

    weights = outputs["expert_weights"][0].cpu().numpy()  # [seq_len, 3]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(weights.T, aspect='auto', cmap='Blues')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Expert r=8', 'Expert r=16', 'Expert r=32'])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im, label='Weight')
    plt.title('Expert Routing per Token')
    plt.tight_layout()
    plt.savefig('routing_visualization.png')
```

---

## Amélioration 7: Benchmark Systématique

### Script de benchmark

```bash
#!/bin/bash
# benchmark_experiments.sh

# Experiment 1: Load balancing sweep
for lb in 0.1 0.01 0.001 0.0001; do
    python train.py \
        --config configs/gated_full.json \
        --output-dir outputs/gated_lb_${lb} \
        --override model.load_balancing_weight=$lb
done

# Experiment 2: Expert ranks
for ranks in "4,8,16" "8,16,32" "16,32,64"; do
    python train.py \
        --config configs/gated_full.json \
        --output-dir outputs/gated_ranks_${ranks} \
        --override model.expert_ranks="[$ranks]"
done

# Experiment 3: Top-K routing
for k in 1 2 3; do
    python train.py \
        --config configs/gated_full.json \
        --output-dir outputs/gated_topk_${k} \
        --override model.use_top_k=true \
        --override model.top_k=$k
done
```

---

## Plan d'Action Recommandé

### Phase 1: Quick Wins (1-2 heures)
1. Créer config `gated_low_lb.json` avec `load_balancing_weight=0.001`
2. Relancer entraînement
3. Comparer les routing patterns

### Phase 2: Dataset (2-4 heures)
1. Configurer un dataset multi-domaine (Pile subset)
2. Entraîner baseline et gated
3. Analyser si le gating s'adapte aux domaines

### Phase 3: Architecture (4-8 heures)
1. Implémenter top-k routing
2. Tester différentes configurations d'experts
3. Créer visualisations

### Phase 4: Publication (variable)
1. Compiler tous les résultats
2. Créer figures publication-ready
3. Rédiger rapport/article

---

## Résultats Attendus

Si les améliorations fonctionnent, on devrait observer:

1. **Load balancing réduit**: Distribution non-uniforme des experts
2. **Dataset multi-domaine**: >2% d'amélioration vs baseline
3. **Top-K routing**: Réduction du temps d'entraînement avec performance similaire
4. **Visualisation**: Patterns de routing interprétables (ex: expert lourd pour tokens rares)

---

*Document créé le 5 décembre 2025*
