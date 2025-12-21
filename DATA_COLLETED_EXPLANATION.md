# Explication des Données Collectées

Ce document explique tous les fichiers générés par le système d'évaluation du DDA et ce qu'ils contiennent.

## Structure des Fichiers

Toutes les données sont sauvegardées dans le dossier `evaluation_data/`.

---

## FICHIERS CONSOLIDÉS (Tous les joueurs ensemble)

### 1. `all_metrics.json`

**Description :** Fichier JSON contenant toutes les métriques calculées de tous les joueurs et tous les matches.

**Structure :** Liste d'objets JSON, un par match.

**Chaque entrée contient :**
```json
{
  "player_id": 6,                    // ID du joueur (auto-incrémenté)
  "match_number": 1,                  // Numéro du match (1 ou 2)
  "condition": "without_dda",         // "with_dda" ou "without_dda"
  "timestamp": "2025-12-21T15:02:04", // Date et heure du match
  "match_duration": 45.2,             // Durée du match en secondes
  
  // MÉTRIQUES DE SURVIE
  "survival_time": 45.2,              // Temps de survie (identique à match_duration)
  "final_score": 120,                 // Score final obtenu
  
  // MÉTRIQUES DE SANTÉ (Flow Zone)
  "average_health": 65.5,             // Santé moyenne (0-100%)
  "health_variance": 12.3,            // Écart-type de la santé (stabilité)
  "min_health": 30,                   // Santé minimum atteinte
  "time_in_flow_zone_percent": 75.0,  // % de temps avec santé 30-70%
  "time_in_danger_percent": 5.0,      // % de temps avec santé < 30%
  "time_in_safety_percent": 20.0,     // % de temps avec santé > 70%
  
  // MÉTRIQUES DE PERFORMANCE
  "total_kills": 12,                  // Nombre total de zombies tués
  "kill_rate_per_minute": 15.9,      // Taux de kills (kills/minute)
  "shooting_accuracy": 72.5,          // Précision de tir (%)
  "total_damage_dealt": 450,          // Total de dégâts infligés
  "total_damage_taken": 80,           // Total de dégâts reçus
  
  // MÉTRIQUES DE DIFFICULTÉ
  "average_zombies_alive": 6.2,      // Nombre moyen de zombies vivants
  "peak_zombies_alive": 12,           // Pic maximum de zombies simultanés
  
  // MÉTRIQUES D'ENGAGEMENT
  "total_pickups_collected": 3,       // Nombre de pickups collectés
  "total_distance_moved": 2500.5,     // Distance totale parcourue
  "average_movement_speed": 55.3,     // Vitesse moyenne de déplacement
  
  // MÉTRIQUES DDA (seulement si condition == "with_dda")
  "total_dda_adjustments": 9,         // Nombre d'ajustements DDA
  "dda_adjustment_frequency": 12.0,   // Fréquence d'ajustements (par minute)
  "dda_action_distribution": {        // Distribution des actions DDA (%)
    "Do Nothing": 55.6,
    "Slightly Easier": 22.2,
    "Much Easier": 11.1,
    "Slightly Harder": 11.1,
    "Much Harder": 0.0
  },
  "dda_oscillations": 2,              // Nombre d'oscillations (changements de direction)
  "dda_oscillation_percent": 22.2     // % d'oscillations
}
```

**Utilisation :** Analyse statistique, comparaisons entre conditions, calculs d'amélioration.

---

### 2. `all_frames.json`

**Description :** Fichier JSON contenant toutes les données frame-by-frame de tous les joueurs et tous les matches.

**Structure :** Liste d'objets JSON, un par match.

**Chaque entrée contient :**
```json
{
  "player_id": 6,
  "match_number": 1,
  "condition": "without_dda",
  "timestamp": "2025-12-21T15:02:04",
  "frames": [
    {
      "time": 0.0,                    // Temps écoulé depuis le début (secondes)
      "player_health": 100,            // Santé du joueur (0-100)
      "zombies_alive": 0,              // Nombre de zombies vivants
      "player_x": 425,                 // Position X du joueur
      "player_y": 325,                 // Position Y du joueur
      "score": 0                       // Score actuel
    },
    {
      "time": 1.0,                    // Données enregistrées chaque seconde
      "player_health": 95,
      "zombies_alive": 2,
      "player_x": 430,
      "player_y": 320,
      "score": 10
    },
    // ... une entrée par seconde de jeu
  ]
}
```

**Utilisation :** 
- Analyse temporelle détaillée
- Graphiques d'évolution de la santé au fil du temps
- Analyse des patterns de mouvement
- Reconstruction complète de la session

**Note :** Les données sont enregistrées **chaque seconde** pendant le match.

---

### 3. `all_answers.json`

**Description :** Fichier JSON contenant toutes les réponses aux questions subjectives de tous les joueurs.

**Structure :** Liste d'objets JSON, un par question (2 par joueur).

**Chaque entrée contient :**
```json
{
  "player_id": 6,
  "match_number": 1,                  // Match 1 ou 2
  "condition": "without_dda",         // Condition de ce match
  "enjoyment": 3,                     // Plaisir (1-5)
  "difficulty": 4,                    // Difficulté (1-5, 3=équilibré)
  "timestamp": "2025-12-21T15:02:04"  // Date et heure de la réponse
}
```

**Questions posées :**
1. **Plaisir** : "À quel point avez-vous apprécié ce match ?" (1-5)
2. **Difficulté** : "Le jeu était-il trop facile, trop difficile, ou équilibré ?" (1-5)

**Utilisation :** Analyse des réponses subjectives, corrélations avec métriques objectives.

---

### 4. `all_preferences.json`

**Description :** Fichier JSON contenant toutes les préférences finales de tous les joueurs.

**Structure :** Liste d'objets JSON, un par joueur.

**Chaque entrée contient :**
```json
{
  "player_id": 6,
  "match1_condition": "without_dda",  // Condition du Match 1
  "match2_condition": "with_dda",    // Condition du Match 2
  "preference": "match2",            // "match1", "match2", ou "no_preference"
  "preferred_condition": "with_dda", // Condition préférée (déduite)
  "timestamp": "2025-12-21T15:04:17" // Date et heure de la réponse
}
```

**Question posée :** "Quelle version avez-vous préférée ?"

**Utilisation :** 
- Analyse de préférence globale
- Test statistique (Chi-carré) pour voir si le DDA est préféré
- Corrélation avec les métriques objectives

---

## FICHIERS DE CONFIGURATION

### 5. `player_counter.json`

**Description :** Fichier JSON qui garde trace du prochain ID de joueur à assigner.

**Structure :**
```json
{
  "next_id": 7  // Prochain ID à assigner (auto-incrémenté)
}
```

**Utilisation :** Gestion automatique des IDs de joueurs. Ne pas modifier manuellement.

---

## FICHIERS INDIVIDUELS (Backup - Optionnels)

Ces fichiers sont créés pour chaque joueur et chaque match comme backup, mais les données consolidées dans `all_*.json` sont prioritaires.

### Format : `{player_id}_match{match_number}_{condition}_metrics.json`
**Exemple :** `3_match1_without_dda_metrics.json`

**Contenu :** Même structure que les entrées dans `all_metrics.json`, mais pour un seul match.

**Utilisation :** Backup individuel, peut être supprimé si les données consolidées sont suffisantes.

---

### Format : `{player_id}_match{match_number}_{condition}_metrics.csv`
**Exemple :** `3_match1_without_dda_metrics.csv`

**Contenu :** Version CSV des métriques (plus facile à ouvrir dans Excel).

**Format CSV :**
```csv
Metric,Value
survival_time,14.528274
final_score,40
average_health,98.0
...
```

**Utilisation :** Analyse rapide dans Excel/Google Sheets, peut être supprimé si JSON est suffisant.

---

### Format : `{player_id}_match{match_number}_{condition}_frames.json`
**Exemple :** `3_match1_without_dda_frames.json`

**Contenu :** Même structure que les entrées dans `all_frames.json`, mais pour un seul match.

**Utilisation :** Backup individuel, peut être supprimé si les données consolidées sont suffisantes.

---

## Résumé des Fichiers

### Fichiers Essentiels (à conserver) :
1. ✅ **`all_metrics.json`** - Toutes les métriques calculées
2. ✅ **`all_frames.json`** - Toutes les données frame-by-frame
3. ✅ **`all_answers.json`** - Toutes les réponses subjectives
4. ✅ **`all_preferences.json`** - Toutes les préférences
5. ✅ **`player_counter.json`** - Compteur d'IDs

### Fichiers de Backup (optionnels, peuvent être supprimés) :
- `{player_id}_match{match_number}_{condition}_metrics.json`
- `{player_id}_match{match_number}_{condition}_metrics.csv`
- `{player_id}_match{match_number}_{condition}_frames.json`

---

## Comment Utiliser les Données

### Pour l'Analyse Statistique :

1. **Charger `all_metrics.json`** pour comparer les métriques entre conditions
2. **Charger `all_answers.json`** pour analyser les réponses subjectives
3. **Charger `all_preferences.json`** pour analyser les préférences

### Pour l'Analyse Temporelle :

1. **Charger `all_frames.json`** pour voir l'évolution de la santé, du nombre de zombies, etc. au fil du temps
2. Créer des graphiques temporels pour chaque joueur

### Exemple d'Analyse Python :

```python
import json

# Charger les métriques
with open('evaluation_data/all_metrics.json', 'r') as f:
    all_metrics = json.load(f)

# Séparer par condition
with_dda = [m for m in all_metrics if m['condition'] == 'with_dda']
without_dda = [m for m in all_metrics if m['condition'] == 'without_dda']

# Calculer moyennes
avg_survival_with = sum(m['survival_time'] for m in with_dda) / len(with_dda)
avg_survival_without = sum(m['survival_time'] for m in without_dda) / len(without_dda)

print(f"Survie moyenne avec DDA: {avg_survival_with:.2f}s")
print(f"Survie moyenne sans DDA: {avg_survival_without:.2f}s")
```

---

## Notes Importantes

1. **Tous les fichiers sont en JSON** (sauf les CSV de backup) pour faciliter l'analyse programmatique
2. **Les timestamps sont en format ISO 8601** (ex: "2025-12-21T15:02:04.740517")
3. **Les données sont consolidées** : tous les joueurs sont dans les mêmes fichiers
4. **Les IDs sont auto-incrémentés** : commencent à 1 et augmentent automatiquement
5. **Les données frame-by-frame** sont enregistrées **chaque seconde** pendant le match
6. **Les métriques sont calculées** après chaque match à partir des données frame-by-frame

---

## Structure Recommandée pour l'Analyse

```
evaluation_data/
├── all_metrics.json          ← PRINCIPAL : Toutes les métriques
├── all_frames.json           ← PRINCIPAL : Toutes les données temporelles
├── all_answers.json          ← PRINCIPAL : Toutes les réponses subjectives
├── all_preferences.json      ← PRINCIPAL : Toutes les préférences
├── player_counter.json       ← Configuration (ne pas modifier)
└── [fichiers individuels]    ← Backups (optionnels, peuvent être supprimés)
```

---

## Prochaines Étapes

Après avoir collecté des données de plusieurs joueurs, utilisez `analyze_metrics.py` pour générer un rapport d'analyse automatique :

```bash
python3 analyze_metrics.py
```

Cela génère :
- `evaluation_report.txt` : Rapport textuel avec statistiques
- `evaluation_report.json` : Données structurées pour analyse approfondie

