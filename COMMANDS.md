# Guide des Commandes - RL_PROJECT

Ce document liste toutes les commandes disponibles pour chaque script du projet.

---

## 📋 Table des Matières

1. [Jeu Principal](#1-jeu-principal-mainpy)
2. [Jouer avec DDA](#2-jouer-avec-dda-play_with_ddapy)
3. [Tester DDA](#3-tester-dda-test_ddapy)
4. [Entraîner PPO Agent](#4-entraîner-ppo-agent-scriptstrain_ppopy)
5. [Entraîner DDA Agent](#5-entraîner-dda-agent-scriptstrain_ddapy)
6. [Tester PPO Agent](#6-tester-ppo-agent-scriptstest_agentpy)
7. [Session de Test d'Évaluation](#7-session-de-test-dévaluation-scriptstest_sessionpy)
8. [Analyser les Métriques](#8-analyser-les-métriques-utilsanalyze_metricspy)

---

## 1. Jeu Principal (`main.py`)

Jouer au jeu zombie shooter en mode manuel ou avec un agent PPO entraîné.

### Commandes de Base

```bash
# Mode manuel (vous jouez avec clavier/souris)
python main.py --mode manual

# Mode agent (PPO agent joue automatiquement)
python main.py --mode agent

# Mode agent avec modèle spécifique
python main.py --mode agent --model checkpoints/best_model.pth
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--mode` | Mode de jeu: `manual` ou `agent` | `manual` |
| `--model` | Chemin vers le modèle PPO entraîné (pour mode agent) | `checkpoints/best_model.pth` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Jouer manuellement
python main.py

# Regarder l'agent PPO jouer
python main.py --mode agent

# Utiliser un modèle spécifique
python main.py --mode agent --model checkpoints/checkpoint_1001472.pth
```

### Contrôles (Mode Manuel)

- **WASD** ou **Flèches** : Déplacer le joueur
- **Souris** : Viser
- **Clic gauche** : Tirer
- **1/2/3** : Changer d'arme (Pistolet/Mitrailleuse/Couteau)
- **R** : Redémarrer
- **ESC** : Quitter

---

## 2. Jouer avec DDA (`play_with_dda.py`)

Jouer au jeu avec le système DDA (Dynamic Difficulty Adjustment) activé. Le DDA ajuste automatiquement la difficulté toutes les 5 secondes.

### Commandes de Base

```bash
# Jouer avec DDA (modèle par défaut)
python play_with_dda.py

# Jouer avec un modèle DDA spécifique
python play_with_dda.py --model checkpoints/dda/best_dda_model.pth
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--model` | Chemin vers le modèle DDA | `checkpoints/dda/best_dda_model.pth` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Jouer avec DDA (modèle par défaut)
python play_with_dda.py

# Utiliser un checkpoint spécifique
python play_with_dda.py --model checkpoints/dda/checkpoint_50000.pth
```

### Affichage DDA

Le jeu affiche en haut à droite :
- **DDA Action** : Décision du DDA (Do Nothing, Much Easier, Slightly Harder, etc.)
- **Zombie Speed** : Vitesse actuelle des zombies
- **Spawn Rate** : Taux d'apparition des zombies

### Contrôles

Identiques au mode manuel de `main.py` :
- **WASD** ou **Flèches** : Déplacer
- **Souris** : Viser
- **Clic gauche** : Tirer
- **1/2/3** : Changer d'arme
- **R** : Redémarrer
- **ESC** : Quitter

---

## 3. Tester DDA (`test_dda.py`)

Tester le système DDA en mode manuel (vous jouez) ou en mode agent (PPO agent joue).

### Commandes de Base

```bash
# Mode manuel (vous jouez)
python test_dda.py --mode manual

# Mode agent (PPO agent joue)
python test_dda.py --mode agent

# Mode agent avec modèles spécifiques
python test_dda.py --mode agent \
    --dda-model checkpoints/dda/best_dda_model.pth \
    --player-model checkpoints/best_model.pth
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--mode` | Mode de test: `manual` ou `agent` | `manual` |
| `--dda-model` | Chemin vers le modèle DDA | `checkpoints/dda/best_dda_model.pth` |
| `--player-model` | Chemin vers le modèle PPO (pour mode agent) | `checkpoints/best_model.pth` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Mode manuel (appelle play_with_dda.py)
python test_dda.py --mode manual

# Mode agent avec modèles par défaut
python test_dda.py --mode agent

# Mode agent avec modèles spécifiques
python test_dda.py --mode agent \
    --dda-model checkpoints/dda/best_dda_model.pth \
    --player-model checkpoints/best_model.pth
```

### Différence avec `play_with_dda.py`

- **`play_with_dda.py`** : Mode manuel uniquement, interface simple
- **`test_dda.py`** : Deux modes (manuel + agent), permet de tester le DDA avec un agent PPO

---

## 4. Entraîner PPO Agent (`scripts/train_ppo.py`)

Entraîner un agent PPO pour jouer au jeu zombie shooter.

### Commandes de Base

```bash
# Entraînement de base (1,000,000 timesteps)
python scripts/train_ppo.py

# Entraînement avec rendu (visualisation)
python scripts/train_ppo.py --render

# Reprendre depuis un checkpoint
python scripts/train_ppo.py --checkpoint checkpoints/checkpoint_1001472.pth
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--timesteps` | Nombre total de timesteps d'entraînement | `1000000` |
| `--render` | Afficher le jeu pendant l'entraînement | `False` |
| `--checkpoint` | Chemin vers un checkpoint pour reprendre l'entraînement | `None` |
| `--device` | Device à utiliser: `cuda` ou `cpu` | `cpu` |
| `--lr` | Learning rate | `3e-4` |
| `--batch-size` | Taille du batch | `64` |
| `--n-steps` | Nombre de steps par rollout | `2048` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Entraînement rapide (100k timesteps)
python scripts/train_ppo.py --timesteps 100000

# Entraînement avec GPU
python scripts/train_ppo.py --device cuda

# Entraînement avec rendu (plus lent)
python scripts/train_ppo.py --render --timesteps 500000

# Reprendre l'entraînement
python scripts/train_ppo.py --checkpoint checkpoints/checkpoint_1001472.pth --timesteps 2000000

# Entraînement avec hyperparamètres personnalisés
python scripts/train_ppo.py --lr 1e-4 --batch-size 128 --n-steps 4096
```

### Fichiers Générés

- **Modèles sauvegardés** : `checkpoints/best_model.pth`, `checkpoints/checkpoint_*.pth`
- **Logs** : `logs/training_YYYYMMDD_HHMMSS.log`

---

## 5. Entraîner DDA Agent (`scripts/train_dda.py`)

Entraîner un agent DDA pour ajuster dynamiquement la difficulté du jeu.

### Commandes de Base

```bash
# Entraînement de base
python scripts/train_dda.py

# Entraînement avec rendu
python scripts/train_dda.py --render

# Reprendre depuis un checkpoint DDA
python scripts/train_dda.py --checkpoint checkpoints/dda/checkpoint_50000.pth
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--player-model` | Chemin vers le modèle PPO du joueur | `checkpoints/best_model.pth` |
| `--checkpoint` | Chemin vers un checkpoint DDA pour reprendre | `None` |
| `--render` | Afficher le jeu pendant l'entraînement | `False` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Entraînement de base
python scripts/train_dda.py

# Utiliser un modèle joueur spécifique
python scripts/train_dda.py --player-model checkpoints/best_model.pth

# Entraînement avec rendu (visualisation)
python scripts/train_dda.py --render

# Reprendre l'entraînement DDA
python scripts/train_dda.py --checkpoint checkpoints/dda/checkpoint_50000.pth
```

### Prérequis

- Un modèle PPO entraîné doit exister (généré par `scripts/train_ppo.py`)
- Le modèle par défaut est `checkpoints/best_model.pth`

### Fichiers Générés

- **Modèles DDA** : `checkpoints/dda/best_dda_model.pth`, `checkpoints/dda/checkpoint_*.pth`
- **Logs** : `logs/training_YYYYMMDD_HHMMSS.log`

---

## 6. Tester PPO Agent (`scripts/test_agent.py`)

Tester les performances d'un agent PPO entraîné.

### Commandes de Base

```bash
# Test de base (10 épisodes)
python scripts/test_agent.py --model checkpoints/best_model.pth

# Test avec plus d'épisodes
python scripts/test_agent.py --model checkpoints/best_model.pth --episodes 50

# Test sans rendu (plus rapide)
python scripts/test_agent.py --model checkpoints/best_model.pth --no-render
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--model` | Chemin vers le modèle à tester (requis) | - |
| `--episodes` | Nombre d'épisodes à tester | `10` |
| `--no-render` | Désactiver le rendu (plus rapide) | `False` |
| `--stochastic` | Utiliser une politique stochastique (défaut: déterministe) | `False` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Test rapide (10 épisodes, avec rendu)
python scripts/test_agent.py --model checkpoints/best_model.pth

# Test approfondi (100 épisodes, sans rendu)
python scripts/test_agent.py --model checkpoints/best_model.pth --episodes 100 --no-render

# Test avec politique stochastique
python scripts/test_agent.py --model checkpoints/best_model.pth --stochastic

# Tester un checkpoint spécifique
python scripts/test_agent.py --model checkpoints/checkpoint_1001472.pth --episodes 20
```

### Statistiques Affichées

- **Average Reward** : Récompense moyenne
- **Average Length** : Longueur moyenne des épisodes
- **Average Kills** : Nombre moyen de zombies tués
- **Best/Worst Reward** : Meilleure et pire récompense

---

## 7. Session de Test d'Évaluation (`scripts/test_session.py`)

Gérer des sessions de test pour évaluer le DDA. Chaque joueur joue 2 matches (un avec DDA, un sans DDA).

### Commandes de Base

```bash
# Lancer une session de test
python scripts/test_session.py
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| Aucune option CLI | Le script gère tout automatiquement | - |

### Fonctionnement

1. **Attribution automatique d'ID** : Le script assigne automatiquement un ID de joueur
2. **Randomisation** : L'ordre des matches (avec/sans DDA) est randomisé
3. **Deux matches** :
   - Match 1 : Avec ou sans DDA (aléatoire)
   - Match 2 : L'autre condition
4. **Questions** : Après chaque match, le joueur répond à des questions subjectives
5. **Préférence finale** : À la fin, le joueur indique sa préférence

### Questions Posées

**Après chaque match :**
1. "À quel point avez-vous apprécié ce match ?" (1-5)
2. "Le jeu était-il trop facile, trop difficile, ou équilibré ?" (1-5)

**Après les deux matches :**
3. "Quelle version avez-vous préférée ?" (Match 1 / Match 2 / Aucune préférence)

### Données Collectées

Toutes les données sont sauvegardées dans `data/evaluation/` :
- `all_metrics.json` : Toutes les métriques calculées
- `all_frames.json` : Données frame-by-frame
- `all_answers.json` : Réponses aux questions subjectives
- `all_preferences.json` : Préférences finales

### Contrôles Pendant le Jeu

- **WASD** ou **Flèches** : Déplacer
- **Souris** : Viser
- **Clic gauche** : Tirer
- **1/2/3** : Changer d'arme
- **ESC** : Continuer après la mort

### Notes Importantes

- Le joueur ne voit pas s'il joue avec ou sans DDA (test en aveugle)
- Chaque match se termine à la mort du joueur
- Les données sont automatiquement sauvegardées

---

## 8. Analyser les Métriques (`utils/analyze_metrics.py`)

Analyser les métriques collectées lors des sessions de test d'évaluation.

### Commandes de Base

```bash
# Analyse avec répertoire par défaut
python utils/analyze_metrics.py

# Analyse avec répertoire spécifique
python utils/analyze_metrics.py --data-dir data/evaluation

# Générer un rapport avec nom personnalisé
python utils/analyze_metrics.py --output mon_rapport.txt
```

### Options Disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--data-dir` | Répertoire contenant les données d'évaluation | `data/evaluation` |
| `--output` | Fichier de sortie pour le rapport | `evaluation_report.txt` |
| `--help` | Afficher l'aide | - |

### Exemples

```bash
# Analyse standard
python utils/analyze_metrics.py

# Analyse avec répertoire personnalisé
python utils/analyze_metrics.py --data-dir data/evaluation

# Générer un rapport avec nom spécifique
python utils/analyze_metrics.py --output rapport_dda_2024.txt

# Analyse complète
python utils/analyze_metrics.py --data-dir data/evaluation --output rapport_complet.txt
```

### Rapport Généré

Le script génère un rapport contenant :
- Statistiques descriptives pour chaque métrique
- Comparaisons entre conditions (avec/sans DDA)
- Tests statistiques (si applicable)
- Visualisations (si applicable)

---

## 🚀 Workflow Recommandé

### 1. Entraîner l'Agent PPO

```bash
# Entraîner l'agent PPO
python scripts/train_ppo.py --timesteps 1000000

# Tester l'agent
python scripts/test_agent.py --model checkpoints/best_model.pth --episodes 10
```

### 2. Entraîner l'Agent DDA

```bash
# Entraîner le DDA (nécessite un agent PPO entraîné)
python scripts/train_dda.py

# Tester le DDA
python test_dda.py --mode agent
```

### 3. Jouer avec le DDA

```bash
# Jouer manuellement avec DDA
python play_with_dda.py

# Ou via test_dda.py
python test_dda.py --mode manual
```

### 4. Évaluer le DDA

```bash
# Lancer des sessions de test
python scripts/test_session.py

# Analyser les résultats
python utils/analyze_metrics.py
```

---

## 📁 Structure des Fichiers

### Modèles Sauvegardés

```
checkpoints/
├── best_model.pth              # Meilleur modèle PPO
├── checkpoint_*.pth            # Checkpoints PPO
└── dda/
    ├── best_dda_model.pth     # Meilleur modèle DDA
    └── checkpoint_*.pth       # Checkpoints DDA
```

### Données d'Évaluation

```
data/evaluation/
├── all_metrics.json           # Toutes les métriques
├── all_frames.json            # Données frame-by-frame
├── all_answers.json           # Réponses subjectives
├── all_preferences.json       # Préférences finales
└── player_counter.json        # Compteur d'IDs
```

### Logs

```
logs/
└── training_YYYYMMDD_HHMMSS.log  # Logs d'entraînement
```

---

## 🔧 Environnement Conda

Toutes les commandes doivent être exécutées dans l'environnement conda `rl_masror` :

```bash
# Activer l'environnement
conda activate rl_masror

# Ou utiliser conda run
conda run -n rl_masror python script.py
```

---

## ❓ Aide et Support

Pour obtenir de l'aide sur n'importe quel script :

```bash
python script.py --help
```

---

## 📝 Notes

- Tous les chemins de modèles sont relatifs à la racine du projet
- Les scripts dans `scripts/` nécessitent d'être exécutés depuis la racine du projet
- Les données d'évaluation sont automatiquement sauvegardées dans `data/evaluation/`
- Les modèles sont sauvegardés dans `checkpoints/` pendant l'entraînement

---

**Dernière mise à jour** : Après réorganisation du projet

