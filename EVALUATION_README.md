# Guide d'Évaluation du DDA

Ce guide explique comment utiliser le système d'évaluation pour tester l'impact du DDA sur l'expérience de jeu.

## Structure du Test

Chaque joueur joue **exactement 2 matches** :
- **Match 1** : Avec DDA OU Sans DDA (ordre randomisé)
- **Match 2** : L'autre condition

Chaque match se termine par la mort du joueur.

## Fichiers Créés

### 1. `metrics_collector.py`
Système de collecte automatique de toutes les métriques définies dans `DDA_EVALUATION_METRICS.md`.

### 2. `test_session.py`
Script principal pour gérer les sessions de test :
- Cache les informations sur les vagues et le DDA
- Collecte automatiquement les métriques
- Pose 3 questions simples après chaque match

### 3. `analyze_metrics.py`
Script pour analyser les métriques collectées et générer un rapport.

## Utilisation

### 1. Lancer une Session de Test

```bash
python test_session.py
```

Le script va :
1. **Assigner automatiquement un ID de joueur** (incrémenté à chaque session)
2. Randomiser l'ordre (DDA en premier ou en second)
3. Lancer le Match 1
4. Poser 2 questions après le Match 1
5. Lancer le Match 2
6. Poser 2 questions après le Match 2
7. Poser la question de préférence finale
8. **Ajouter toutes les données aux fichiers consolidés**

**Important :** 
- Les informations sur les vagues et le DDA sont **cachées** pendant le gameplay pour éviter les biais
- L'ID est géré automatiquement - pas besoin de le saisir
- Toutes les données sont ajoutées aux mêmes fichiers consolidés

### 2. Analyser les Résultats

Après avoir collecté des données de plusieurs joueurs :

```bash
python analyze_metrics.py
```

Cela génère :
- `evaluation_report.txt` : Rapport textuel avec statistiques
- `evaluation_report.json` : Données structurées pour analyse approfondie

## Données Collectées

Toutes les données sont sauvegardées dans le dossier `evaluation_data/` :

### Fichiers Consolidés (tous les joueurs) :
- `all_metrics.json` : **Toutes les métriques de tous les joueurs** (liste JSON)
- `all_answers.json` : **Toutes les réponses subjectives de tous les joueurs** (liste JSON)
- `all_preferences.json` : **Toutes les préférences de tous les joueurs** (liste JSON)
- `player_counter.json` : Compteur pour gérer les IDs automatiquement

### Fichiers Individuels (backup, optionnel) :
- `{player_id}_match{1|2}_{condition}_metrics.json` : Métriques calculées (backup)
- `{player_id}_match{1|2}_{condition}_metrics.csv` : Métriques en format CSV (backup)
- `{player_id}_match{1|2}_{condition}_frames.json` : Données frame-by-frame (backup)

## Métriques Collectées

### Métriques Automatiques (Top 6) :
1. **Durée de Survie** - Temps total avant la mort
2. **Temps dans Flow Zone** - % de temps avec santé 30-70%
3. **Santé Moyenne** - Santé moyenne pendant le match
4. **Temps en Danger** - % de temps avec santé < 30%
5. **Nombre de Zombies Tués** - Total de kills
6. **Variance de la Santé** - Stabilité de l'expérience

### Questions Subjectives (3 questions) :
1. **Plaisir** : "À quel point avez-vous apprécié ce match ?" (1-5)
2. **Difficulté** : "Le jeu était-il trop facile, trop difficile, ou équilibré ?" (1-5, 3=équilibré)
3. **Préférence** : "Quelle version avez-vous préférée ?" (après les deux matches)

## Informations Cachées

Pour éviter les biais, les informations suivantes sont **cachées** pendant le gameplay :
- ❌ **Wave** (vague) - Non affichée dans l'UI
- ❌ **DDA Actions** - Aucune indication visible du DDA
- ❌ **Paramètres de difficulté** - Non visibles

Les joueurs voient seulement :
- ✅ Score
- ✅ Zombies tués
- ✅ Santé
- ✅ Arme actuelle
- ✅ Munitions (si machinegun)

## Exemple de Session

```
Player ID: player_001
Match 1: with_dda
Match 2: without_dda

[Match 1 joué...]
Questions Match 1:
- Plaisir: 4/5
- Difficulté: 3/5

[Match 2 joué...]
Questions Match 2:
- Plaisir: 3/5
- Difficulté: 4/5

Préférence: Le premier match (with_dda)
```

## Analyse des Résultats

**Note :** L'analyse se fait après la collecte de données. Utilisez les fichiers consolidés :
- `all_metrics.json` : Pour analyser les métriques automatiques
- `all_answers.json` : Pour analyser les réponses subjectives
- `all_preferences.json` : Pour analyser les préférences

Vous pouvez créer votre propre script d'analyse ou utiliser `analyze_metrics.py` (à adapter pour lire les fichiers consolidés).

## Notes Importantes

1. **ID automatique** : L'ID du joueur est assigné automatiquement et incrémenté à chaque session
2. **Ordre randomisé** : L'ordre (DDA en premier ou en second) est randomisé pour chaque joueur
3. **Blind testing** : Les joueurs ne savent pas quelle version ils jouent
4. **Questions immédiates** : Les questions sont posées immédiatement après chaque match
5. **Une mort par match** : Chaque match se termine par la mort du joueur
6. **Données consolidées** : Toutes les données sont ajoutées aux mêmes fichiers pour faciliter l'analyse

## Troubleshooting

### Le script ne trouve pas le modèle DDA
Assurez-vous que le fichier `checkpoints/dda/best_dda_model.pth` existe. Si vous avez un autre chemin, modifiez `test_session.py`.

### Erreur lors de la collecte de métriques
Vérifiez que le dossier `evaluation_data/` existe et est accessible en écriture.

### Pas de données pour l'analyse
Assurez-vous d'avoir collecté des données d'au moins un joueur avant de lancer l'analyse.

