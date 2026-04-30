# Audit Report: Gated LoRA Implementation vs Specification

## Executive Summary

**Conclusion: L'implémentation actuelle présente plusieurs écarts significatifs par rapport à la spécification originale, ce qui explique les résultats décevants (pas de spécialisation des experts, overfitting des modèles gated).**

### Résultats d'analyse du routing

| Experiment | Specialization Score | Observation |
|------------|---------------------|-------------|
| Exp2 (2 experts) | 0.0036 | **Très faible** - Quasi aucune différenciation par tâche |
| Exp3 (3 experts) | 0.0034 | **Très faible** - Idem |
| Exp4 (+ Load Balancing) | 0.0051 | **Très faible** - Marginal improvement |

Un score de spécialisation < 0.01 signifie que les experts sont utilisés de manière quasi-identique pour toutes les tâches. **Le gating n'a pas appris à router différemment selon les tâches.**

---

## Écarts Critiques avec la Spécification

### 1. **Architecture du Gating - MAJEUR**

**Spécification originale:**
> "Per head × layer: Input is current token embedding (768) and **layer index**; architecture employs an MLP (768→256→3, ReLU)"

**Implémentation actuelle:**
```python
# gating_network.py - GatingMLP
self.gate = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),  # hidden_dim → 256
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_experts),
)
```

**Problème:** Le gating ne reçoit PAS le layer index comme entrée. Il utilise uniquement les hidden states. Cela empêche le gating de différencier son comportement selon la profondeur du réseau.

**Fix suggéré:** Ajouter un embedding de layer index ou le concaténer aux hidden states.

---

### 2. **Granularité du Routing - MAJEUR**

**Spécification originale:**
> "Gated LoRA enables each token, **in each layer and attention head**, to dynamically select its update rank"
> "Per-layer, **per-head** gating network"

**Implémentation actuelle:**
- Le gating est **per-layer** seulement, PAS per-head
- Un seul vecteur de weights [batch, seq, num_experts] pour toute la layer
- Tous les attention heads utilisent le même routing

**Impact:** Perte significative d'expressivité. La spécification prévoyait 32 layers × 32 heads = 1024 décisions de routing distinctes par token. L'implémentation n'en fait que 32.

---

### 3. **Application des LoRA - MAJEUR**

**Spécification originale:**
> Architecture montre LoRA appliqué à q_proj, k_proj, v_proj, dense séparément avec routing distinct

**Implémentation actuelle:**
```python
# gated_lora_v2.py line 279-283
lora_delta = expert_pool.get_weighted_output(
    hidden_states,
    module_name=self.target_modules[0],  # PRIMARY MODULE ONLY!
    gate_weights=gate_weights,
)
```

**Problème CRITIQUE:** Seul le premier module target (q_proj) reçoit le LoRA! Les autres modules (k_proj, v_proj, dense) sont ignorés. C'est un BUG majeur.

---

### 4. **Pas d'initialisation SVD**

**Spécification originale:**
> "LoRA Experts (Frozen after SVD Init)"

**Implémentation actuelle:**
```python
# lora_experts.py
def _init_weights(self):
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)
```

Initialisation standard LoRA, pas d'initialisation SVD. Ce n'est pas forcément un problème majeur, mais diffère de la spec.

---

### 5. **Pas de régularisation L1 sur les gates**

**Spécification originale:**
> "Loss incorporates LM loss and **L1 regularization of gate weights** for sparsity"

**Implémentation actuelle:** Utilise uniquement load balancing loss (style Switch Transformer), pas de L1.

---

### 6. **Warmup mal conçu (désactivé)**

**Spécification originale:**
> "Warmup (epoch 1): Only the gating network is trained. LoRA experts frozen, learning rank utility per position."

**Problème identifié et corrigé:** Le warmup freezait les experts mais le gating seul ne peut pas réduire la loss (il n'a pas d'effet sur les outputs sans experts actifs). Warmup désactivé.

---

## Analyse des Résultats

### Pourquoi la spécialisation est nulle?

1. **Bug du module unique:** Seul q_proj est adapté. k_proj, v_proj, dense n'ont pas de LoRA appliqué.

2. **Pas d'information de layer:** Le gating ne sait pas dans quelle layer il opère, donc il apprend un pattern global.

3. **Pas de per-head routing:** Moins d'opportunité de spécialisation fine.

### Pourquoi les gated overfittent?

| Exp | Train Loss | Eval Loss |
|-----|-----------|-----------|
| 1 (Baseline) | 0.8751 | **0.7441** |
| 2 (Gated 2exp) | 0.7934 | 0.7483 |
| 3 (Gated 3exp) | 0.7819 | 0.7519 |
| 4 (+ LoadBal) | 0.8286 | 0.7538 |

Les modèles gated ont plus de paramètres mais performent moins bien en eval. C'est de l'overfitting classique, mais aussi probablement dû au fait que le gating n'apprend pas de spécialisation utile.

---

## Recommandations

### Priorité CRITIQUE (Bugs)

1. **Fixer l'application LoRA à tous les modules**
   ```python
   # Au lieu de:
   lora_delta = expert_pool.get_weighted_output(
       hidden_states,
       module_name=self.target_modules[0],  # BUG!
       ...
   )

   # Faire:
   total_delta = 0
   for module_name in self.target_modules:
       delta = expert_pool.get_weighted_output(
           hidden_states, module_name, gate_weights
       )
       total_delta += delta
   ```

### Priorité HAUTE (Conformité spec)

2. **Ajouter layer index au gating**
   ```python
   # Dans GatingMLP
   self.layer_embedding = nn.Embedding(num_layers, hidden_dim)

   def forward(self, x, layer_idx):
       layer_emb = self.layer_embedding(layer_idx)
       x = x + layer_emb  # ou concatenate
       return self.gate(x)
   ```

3. **Ajouter L1 regularization sur les gates**
   ```python
   l1_loss = gate_weights.abs().mean() * l1_weight
   total_loss = lm_loss + lb_loss + l1_loss
   ```

### Priorité MOYENNE (Amélioration)

4. **Per-head gating** - Plus complexe, nécessite restructuration

5. **SVD initialization** - Optionnel

---

## Conclusion

L'implémentation actuelle a un **bug critique** (seul q_proj reçoit LoRA) et plusieurs **écarts architecturaux** par rapport à la spec originale. Cela explique:

1. **Pas de spécialisation** → Le gating n'a pas assez d'information pour différencier
2. **Overfitting** → Plus de paramètres sans gain utile
3. **Résultats équivalents au baseline** → Le "gating" n'est pas vraiment fonctionnel

**Recommandation:** Corriger le bug du module unique en priorité, puis ajouter le layer index au gating, avant de relancer les expériences.
