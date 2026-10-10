# Prédiction du pays d'un aéroport avec des GNN (GCN, GAT, GraphSAGE)

Projet M2 : classification de nœuds sur le réseau mondial des aéroports (3 363 nœuds, 212 pays).

## Structure du dépôt

```
graphes/airportsAndCoordAndPop.graphml   données (graphe + population + coordonnées)
src/gnn_airports.ipynb                   notebook : exploration, GCN/GAT/GraphSAGE, kNN, 1 seule exécution
src/run_experiments.py                   expériences multi-seeds + ablation (résultats du rapport)
src/make_tables.py                       agrège les seeds (moyenne ± écart-type) et écrit les tableaux LaTeX
results/                                 results.json et tableaux générés par run_experiments.py
```

## Installation

Python 3.12 (le notebook a été exécuté avec 3.12.2).

```bash
pip install torch torch_geometric scikit-learn networkx matplotlib seaborn numpy
```


## Reproduire les résultats du rapport

Depuis le dossier `src/` :

```bash
python run_experiments.py --smoke      # test rapide (1 seed, 5 époques) : vérifie que tout tourne
python run_experiments.py --seeds 10   # expérience complète (voir ci-dessous)
```

Ce que fait le script, pour chaque seed `s` de 0 à 9 :
- nouveau découpage aléatoire train/val/test (70/15/15 %, `torch.randperm` avec un générateur fixé à `s`), **identique pour tous les modèles** (comparaison appariée) ;
- nouvelle initialisation des poids (`torch.manual_seed(s)`) ;
- entraînement de chaque modèle (Adam, perte NLL, arrêt précoce sur la perte de validation) puis mesure de l'accuracy (train/val/test) et du F1 macro (test) ;
- modèles : GCN, GCN amélioré, GAT, GraphSAGE (features d'origine puis enrichies), MLP, kNN, et les variantes de l'ablation.

Sorties dans `results/` : `results.json` (toutes les valeurs par seed), `results_main.tex` (tableau de comparaison, moyenne ± écart-type, meilleure valeur en gras) et `results_ablation.tex` (ablation du modèle final). Les hyperparamètres de chaque modèle sont dans le dictionnaire `CONFIGS` de `run_experiments.py`. Pour régénérer seulement les tableaux : `python make_tables.py ../results/results.json ../results`.

Durée : *40 minutes* (le GAT est le modèle le plus lent).
Les résultats peuvent varier légèrement d'une machine ou d'une version de PyTorch à l'autre (opérations non déterministes).



## Utilisation d'IA générative


- **Modèle :** Claude (Anthropic), identifiant de modèle configuré `claude-sonnet-5-5`, utilisé via l'application Claude / Claude Code.

- **Ce que nous avons demandé à Claude, pourquoi et comment :**
  1. *Suggestions de modèle :* nous lui avons demandé quels modèles pourraient faire mieux que mon GCN (il a proposé GAT et GraphSAGE, plus un kNN comme référence).
  2. *Suggestions de paramètres supplémentaires :* nous lui avons demandé comment améliorer le GCN (couche cachée plus large, connexion résiduelle, dropout, learning rate) et quelles features ajouter (coordonnées 3D, encodages sinusoïdaux, degré).
  3. *Code :* Claude a écrit le code de  `src/run_experiments.py` (10 seeds + ablation) et `src/make_tables.py` (tableaux du rapport). 
  

- **Exemples de prompts (extraits de notre conversation) :**
 
  - « quels recommandation aurais tu pour un autre modele qui pourrais etre meilleur que le gcn ou bien modifier le gcn pour qu'il soit meilleur ? »
  
   - « écris moi une ablation study sur mon modele graphSage »
