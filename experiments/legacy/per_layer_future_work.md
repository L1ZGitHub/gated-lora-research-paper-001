# Future Work & Research Directions

Ce document résume les pistes d'amélioration et les angles de publication envisagés pour la recherche sur Gated LoRA.

**Date de création** : 2024-12-06
**État du code** : Figé (per-layer gating, experiments 1-5)

---

## Table des Matières

1. [Résultats Actuels](#résultats-actuels)
2. [Partie 1 : Améliorations Techniques](#partie-1--améliorations-techniques)
3. [Partie 2 : Changement d'Angle d'Attaque](#partie-2--changement-dangle-dattaque)
4. [Recommandation Finale](#recommandation-finale)

---

## Résultats Actuels

### Performance

| Expérience | Train Loss | Eval Loss | vs Baseline |
|------------|-----------|-----------|-------------|
| Exp1 (Baseline LoRA r=16) | 0.8751 | 0.7441 | — |
| Exp2 (Gated 2 experts) | 1.2014 | 0.7443 | +0.0002 |
| Exp3 (Gated 3 experts) | 1.0741 | **0.7435** | **-0.0006** |
| Exp4 (+Load Balancing) | 1.1389 | **0.7437** | **-0.0004** |
| Exp5 (Top-K) | *en cours* | — | — |

### Spécialisation par Tâche (Couches 30-31)

**Exp2 - Layer 30** :
| Task | E0 (r=8) | E1 (r=16) |
|------|----------|-----------|
| IMDB | **52%** | 48% |
| CoNLL | 6% | **94%** |
| SQuAD | 8% | **92%** |
| WikiText | 31% | 69% |

**Exp3 - Layer 31** :
| Task | E0 (r=8) | E1 (r=16) | E2 (r=32) |
|------|----------|-----------|-----------|
| IMDB | 6% | **46%** | 47% |
| SQuAD | 2% | 30% | **69%** |
| CoNLL | 1% | 5% | **94%** |
| WikiText | 1% | 1% | **97%** |

**Observation clé** : IMDB (sentiment) utilise systématiquement des experts plus petits que les autres tâches.

---

## Partie 1 : Améliorations Techniques

### 1.1 Renforcer la Significance Statistique

**Problème** : Un seul run par expérience ne permet pas de conclusions statistiquement robustes.

**Solution** :
```bash
# Lancer 3-5 runs avec seeds différentes
for seed in 42 123 456 789 1337; do
    python train_v2.py --experiment 3 --seed $seed --output-dir outputs/exp3_seed$seed
done
```

**Effort** : 3-5 jours de calcul
**Impact** : Élevé (essentiel pour publication)

---

### 1.2 Tester sur d'Autres Modèles

**Problème** : Résultats uniquement sur Phi-2 (2.7B params).

**Options** :
- **Llama-2-7B** : Plus grand, architecture différente
- **Mistral-7B** : State-of-the-art open source
- **Phi-3** : Même famille, plus récent

**Script à adapter** :
```python
# Dans train_v2.py, modifier model_name dans config
model_config.model_name = "meta-llama/Llama-2-7b-hf"
model_config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Effort** : 3-4 jours
**Impact** : Très élevé (généralisation des résultats)

---

### 1.3 Datasets Plus Complexes / Spécialisés

**Problème** : Tâches actuelles (SQuAD, IMDB, CoNLL, WikiText) sont relativement simples et similaires.

**Options de datasets** :
- **Code** : CodeSearchNet, HumanEval
- **Math** : GSM8K, MATH
- **Multilingue** : XNLI, mT5 tasks
- **Long context** : NarrativeQA, QuALITY

**Hypothèse** : La spécialisation sera plus marquée sur des tâches très différentes (code vs texte vs math).

**Effort** : 2-3 jours
**Impact** : Moyen-Élevé

---

### 1.4 Mix de Triplets de Tâches

**Idée** : Entraîner sur différentes combinaisons de 3 tâches, analyser les patterns de routing.

**Expériences proposées** :
```
Triplet A: SQuAD + IMDB + WikiText (sans NER)
Triplet B: SQuAD + CoNLL + WikiText (sans sentiment)
Triplet C: IMDB + CoNLL + WikiText (sans QA)
```

**Questions de recherche** :
- Le routing s'adapte-t-il aux tâches présentes ?
- Peut-on prédire le routing optimal ?

**Effort** : 2-3 jours
**Impact** : Moyen

---

### 1.5 Ablations

**Ablations à réaliser** :

| Ablation | Description | Hypothèse |
|----------|-------------|-----------|
| Sans L1 regularization | `l1_gate_weight=0` | Plus d'overfitting, routing moins sparse |
| Sans layer embedding | `use_layer_embedding=False` | Moins de spécialisation par couche |
| Différents expert ranks | `[4,8,16]` vs `[8,16,32]` vs `[16,32,64]` | Impact de la capacité |
| Gating seulement dernières couches | Appliquer gating aux layers 28-31 only | Même effet, moins de params |
| Température du softmax | `temperature=0.5, 1.0, 2.0` | Impact sur la sparsité du routing |

**Effort** : 2-3 jours
**Impact** : Élevé (essentiel pour comprendre les contributions)

---

### 1.6 Per-Head × Per-Layer Gating

**Description** : Actuellement, le gating est per-layer (32 décisions). La spec originale prévoyait per-head × per-layer (32 × 32 = 1024 décisions).

**Avantages** :
- Plus d'expressivité
- Chaque attention head peut utiliser un expert différent
- Plus proche de la spec originale

**Inconvénients** :
- ~32x plus de paramètres de gating
- Plus complexe à implémenter
- Risque d'overfitting

**Fichiers à modifier** :
- `src/models/gating_network.py` : Créer `PerHeadGatingNetwork`
- `src/models/gated_lora_v2.py` : Adapter les hooks pour per-head
- `src/models/lora_experts.py` : Adapter `get_weighted_output` pour per-head

**Effort** : 1-2 semaines
**Impact** : Incertain (risqué)

---

### 1.7 Autres Idées d'Expériences

1. **Analyse de convergence** : Plotter l'évolution du routing pendant le training

2. **Zero-shot routing** : Entraîner sur 3 tâches, tester si le routing se généralise à une 4ème tâche jamais vue

3. **Efficiency analysis** : Comparer FLOPs/mémoire vs baseline pour même performance

4. **Pruning d'experts** : Après training, peut-on retirer un expert sans perte de perf?

5. **Visualisation attention** : Montrer que les experts activent différentes patterns d'attention

6. **Distillation** : Distiller le modèle gated vers un modèle avec routing fixe

---

## Partie 2 : Changement d'Angle d'Attaque

L'angle actuel "Gated LoRA performs better" est faible car le gain de performance est marginal. Voici des angles alternatifs plus prometteurs :

---

### 2.1 📊 "Understanding Routing in Mixture-of-LoRA"

> *"Where and when do LoRA experts specialize? An empirical study"*

**Contribution** : Pas "on fait mieux", mais "on comprend comment ça marche"

**Découvertes à mettre en avant** :
- Spécialisation concentrée dans les couches 30-31
- Sentiment (IMDB) utilise des experts plus petits
- Load balancing force l'équilibre mais préserve la spécialisation locale

**Forces** :
- Nos données suffisent
- Moins besoin de battre une baseline
- Papers d'analyse très cités (ex: "What do Vision Transformers Learn?")

**Cibles** : EMNLP Analysis Track, ACL Findings

---

### 2.2 🧠 "Task Complexity Emerges in Expert Selection"

> *"Gated LoRA learns to allocate capacity based on task difficulty"*

**Contribution** : Le routing apprend implicitement la complexité des tâches

**Argument** :
- IMDB (sentiment binaire) → experts petits (r=8)
- CoNLL (NER structuré) → experts moyens/gros
- SQuAD (QA avec raisonnement) → experts gros (r=32)

**Forces** :
- Angle cognitif/interprétable original
- Connexion avec littérature sur task difficulty
- Testable avec plus de tâches de complexités variées

**Travail additionnel** : Ajouter des tâches de complexités variées (trivial → très difficile)

**Cibles** : ACL, EMNLP

---

### 2.3 🔬 "Layer-wise Specialization in Parameter-Efficient Fine-tuning" ⭐ RECOMMANDÉ

> *"Not all layers need the same LoRA: Evidence from gated routing"*

**Contribution** : Les couches ont des besoins différents en adaptation

**Argument central** :
- Couches 0-29 : routing quasi-uniforme → LoRA standard suffit
- Couches 30-31 : routing très spécialisé → bénéficient du gating

**Implication pratique** : On pourrait appliquer Gated LoRA **seulement aux dernières couches** (moins de params, même effet)

**Forces** :
- Insight actionnable pour la communauté
- Facile à valider avec ablation (gating only last 2-4 layers)
- Titre accrocheur : *"Not All Layers Need the Same LoRA"*

**Travail additionnel** :
1. Ablation : Gated LoRA seulement sur couches 28-31
2. Comparer params/performance avec full gating vs partial gating

**Cibles** : ICLR, NeurIPS

---

### 2.4 🎯 "When Does Mixture-of-Experts Help in LoRA?"

> *"Conditions under which gated routing improves parameter-efficient tuning"*

**Contribution** : Identifier quand MoE-LoRA vaut le coup

**Structure du papier** :
- Multi-task : oui (spécialisation par tâche démontrée)
- Single-task : probablement non (à tester)
- Tâches très différentes : plus de bénéfice (à tester)
- Nombre d'experts : 3 semble optimal (à confirmer)

**Forces** :
- Paper de "negative results" + insights positifs
- Très utile pour practitioners
- Guide quand utiliser/ne pas utiliser

**Travail additionnel** :
1. Single-task experiments
2. Varier le nombre d'experts (2, 3, 4, 5)

**Cibles** : EMNLP Findings, EACL

---

### 2.5 🔄 "Emergence of Functional Specialization in Learned Routing"

> *"Do LoRA experts learn to specialize without explicit supervision?"*

**Contribution** : La spécialisation émerge naturellement

**Argument** :
- Pas de supervision sur "quel expert pour quelle tâche"
- Pourtant le routing converge vers patterns interprétables
- Parallèle avec neurosciences (spécialisation fonctionnelle du cerveau)

**Forces** :
- Angle très original
- Connexion interdisciplinaire (ML + neurosciences)
- Publishable même sans gain de performance

**Travail additionnel** :
1. Visualisations du routing pendant training (émergence progressive)
2. Analyse de stabilité (différents seeds convergent-ils vers même pattern?)

**Cibles** : NeurIPS (ambitieux), ICLR

---

### Comparaison des Angles

| Angle | Originalité | Données Suffisantes | Effort Add. | Cible |
|-------|-------------|---------------------|-------------|-------|
| 2.1 Understanding Routing | ⭐⭐⭐ | ✅ Oui | Faible | EMNLP Analysis |
| 2.2 Task Complexity | ⭐⭐⭐⭐ | ⚠️ +tâches variées | Moyen | ACL/EMNLP |
| **2.3 Layer-wise** | ⭐⭐⭐⭐ | ✅ +ablation | **Faible** | **ICLR/NeurIPS** |
| 2.4 When MoE Helps | ⭐⭐⭐ | ⚠️ +single-task | Moyen | EMNLP Findings |
| 2.5 Emergence | ⭐⭐⭐⭐⭐ | ✅ Oui | Faible | NeurIPS |

---

## Recommandation Finale

### Court terme (1-2 semaines)
1. **Attendre exp5** (Top-K) pour compléter les résultats
2. **Multiple runs** (3 seeds) pour exp 2, 3, 4 → significance statistique
3. **Ablation clé** : Gated LoRA seulement sur couches 28-31

### Moyen terme (2-4 semaines)
4. **Choisir l'angle 2.3** ("Not All Layers Need the Same LoRA")
5. **Tester sur Llama-2-7B** pour généralisation
6. **Rédiger le papier** avec structure :
   - Introduction : "LoRA applique même adaptation à toutes les couches, est-ce optimal?"
   - Méthode : Gated LoRA avec analyse per-layer
   - Résultats : Spécialisation dans couches finales
   - Insight : Appliquer gating partiellement suffit

### Long terme (optionnel)
7. Per-head gating (si résultats layer-wise sont publiés)
8. Extension à d'autres architectures (vision, multimodal)

---

## Structure des Dossiers

```
gated-lora-research/
├── ensicompute_per_layer/     # VERSION FIGÉE - Ne plus modifier
│   ├── src/
│   ├── configs/
│   ├── FUTURE_WORK.md         # Ce fichier
│   └── ...
│
├── ensicompute_ablations/     # À créer pour ablations
│
├── ensicompute_llama/         # À créer pour tests Llama
│
└── paper/                     # À créer pour rédaction
    ├── figures/
    ├── tables/
    └── main.tex
```

---

*Document généré le 2024-12-06*
