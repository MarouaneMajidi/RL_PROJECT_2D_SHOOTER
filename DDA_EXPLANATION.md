# Comment fonctionne le DDA (Dynamic Difficulty Adjustment)

## Vue d'ensemble

Le DDA est un agent d'apprentissage par renforcement qui observe votre performance et ajuste automatiquement la difficulté du jeu pour maintenir un défi équilibré.

---

## 1. CE QUE LE DDA OBSERVE (Inputs)

Toutes les **5 secondes**, le DDA collecte **9 métriques** sur les **10 dernières secondes** :

### Métriques observées :

1. **`player_health_ratio`** (0.0 - 1.0)
   - Santé actuelle du joueur / 100
   - Exemple : 0.5 = 50 HP, 0.3 = 30 HP

2. **`average_damage_taken_last_10s`** (0.0 - 1.0)
   - Dégâts moyens pris par seconde (normalisé)
   - Exemple : 0.8 = beaucoup de dégâts, 0.2 = peu de dégâts

3. **`average_damage_dealt_last_10s`** (0.0 - 1.0)
   - Dégâts moyens infligés par seconde (normalisé)
   - Exemple : 0.9 = vous tuez beaucoup, 0.1 = vous tuez peu

4. **`kill_rate_last_10s`** (0.0 - 1.0)
   - Nombre de zombies tués par seconde (normalisé)
   - Exemple : 0.5 = 1 kill/seconde, 0.0 = aucun kill

5. **`shooting_accuracy_last_10s`** (0.0 - 1.0)
   - Précision de tir (tirs touchés / tirs tirés)
   - Exemple : 0.8 = 80% de précision, 0.3 = 30% de précision

6. **`number_of_zombies_alive`** (0.0 - 1.0)
   - Nombre de zombies vivants (normalisé par max)
   - Exemple : 0.5 = 10 zombies sur 20 max, 1.0 = 20 zombies

7. **`time_since_last_player_hit`** (0.0 - 1.0)
   - Temps depuis le dernier coup reçu (normalisé)
   - Exemple : 0.0 = vient d'être touché, 1.0 = pas touché depuis 5+ secondes

8. **`ammo_ratio`** (0.0 - 1.0)
   - Munitions restantes / max
   - Exemple : 0.5 = 50 munitions sur 100 max

9. **`pickups_collected_last_10s`** (0.0 - 1.0)
   - Pickups collectés par seconde (normalisé)
   - Exemple : 0.3 = collecte active, 0.0 = pas de collecte

**Ces 9 valeurs forment un vecteur d'état** qui représente la situation actuelle du jeu.

---

## 2. COMMENT LE DDA FAIT SES PRÉDICTIONS

### Le modèle RL (Réseau de neurones)

Le DDA utilise un **réseau de neurones** entraîné avec l'algorithme **PPO** (Proximal Policy Optimization).

```
État (9 valeurs) → Réseau de neurones → Action (0-4)
```

### Architecture du réseau :

1. **Couche d'entrée** : 9 neurones (une par métrique)
2. **Couches cachées** : 128 neurones avec ReLU
3. **Couche de sortie** : 5 neurones (une par action possible)

### Processus de décision :

1. Le réseau reçoit le vecteur d'état (9 valeurs)
2. Il calcule des **probabilités** pour chaque action
3. Il choisit l'action avec la **probabilité la plus élevée** (mode déterministe)
4. L'action est retournée : **0, 1, 2, 3, ou 4**

---

## 3. ACTIONS POSSIBLES (Prédictions)

Le DDA peut choisir **5 actions** :

| Action | Nom | Signification | Ajustement |
|--------|-----|---------------|------------|
| **0** | Do Nothing | Aucun changement | 0% |
| **1** | Slightly Easier | Légèrement plus facile | -5% |
| **2** | Much Easier | Beaucoup plus facile | -15% |
| **3** | Slightly Harder | Légèrement plus difficile | +5% |
| **4** | Much Harder | Beaucoup plus difficile | +15% |

---

## 4. RÉSULTATS DANS LE JEU (Effets des actions)

Quand le DDA choisit une action, elle affecte **5 groupes de paramètres** :

### Groupe 1 : Enemy Pressure (Pression ennemie)

**Action "Easier" (-5% ou -15%)** :
- ✅ **ZOMBIE_SPEED** ↓ (zombies plus lents)
- ✅ **SPAWN_RATE** ↑ (spawn plus lent = moins de zombies)
- ✅ **ZOMBIE_ATTACK_COOLDOWN** ↑ (zombies attaquent moins souvent)

**Action "Harder" (+5% ou +15%)** :
- ❌ **ZOMBIE_SPEED** ↑ (zombies plus rapides)
- ❌ **SPAWN_RATE** ↓ (spawn plus rapide = plus de zombies)
- ❌ **ZOMBIE_ATTACK_COOLDOWN** ↓ (zombies attaquent plus souvent)

### Groupe 2 : Enemy Durability (Durabilité ennemie)

**Action "Easier"** :
- ✅ **ZOMBIE_NORMAL_HEALTH** ↓ (zombies normaux ont moins de vie)
- ✅ **ZOMBIE_STRONG_HEALTH** ↓ (zombies forts ont moins de vie)

**Action "Harder"** :
- ❌ **ZOMBIE_NORMAL_HEALTH** ↑ (zombies normaux ont plus de vie)
- ❌ **ZOMBIE_STRONG_HEALTH** ↑ (zombies forts ont plus de vie)

### Groupe 3 : Player Power (Puissance du joueur)

**Action "Easier"** :
- ✅ **PLAYER_SPEED** ↑ (joueur plus rapide)
- ✅ **PISTOL_DAMAGE** ↑ (pistolet fait plus de dégâts)
- ✅ **MACHINEGUN_DAMAGE** ↑ (mitraillette fait plus de dégâts)

**Action "Harder"** :
- ❌ **PLAYER_SPEED** ↓ (joueur plus lent)
- ❌ **PISTOL_DAMAGE** ↓ (pistolet fait moins de dégâts)
- ❌ **MACHINEGUN_DAMAGE** ↓ (mitraillette fait moins de dégâts)

### Groupe 4 : Resource Generosity (Générosité des ressources)

**Action "Easier"** :
- ✅ **HEALTH_PICKUP_DROP_PROBABILITY** ↑ (plus de drops de santé)
- ✅ **MACHINEGUN_PICKUP_DROP_PROBABILITY** ↑ (plus de drops de mitraillette)

**Action "Harder"** :
- ❌ **HEALTH_PICKUP_DROP_PROBABILITY** ↓ (moins de drops de santé)
- ❌ **MACHINEGUN_PICKUP_DROP_PROBABILITY** ↓ (moins de drops de mitraillette)

### Groupe 5 : Combat Tempo (Rythme de combat)

**Action "Easier"** :
- ✅ **PISTOL_COOLDOWN** ↓ (pistolet tire plus vite)
- ✅ **MACHINEGUN_COOLDOWN** ↓ (mitraillette tire plus vite)

**Action "Harder"** :
- ❌ **PISTOL_COOLDOWN** ↑ (pistolet tire plus lentement)
- ❌ **MACHINEGUN_COOLDOWN** ↑ (mitraillette tire plus lentement)

---

## 5. EXEMPLE CONCRET

### Scénario : Joueur en difficulté

**État observé** :
- Health: 0.2 (20 HP) ← **Très bas !**
- Damage taken: 0.8 (beaucoup de dégâts) ← **Trop de pression !**
- Zombies alive: 0.9 (beaucoup de zombies) ← **Submergé !**
- Kill rate: 0.1 (peu de kills) ← **Ne tue pas assez !**

**Prédiction du DDA** :
- Le réseau voit : santé basse + beaucoup de dégâts + beaucoup de zombies
- Il calcule : probabilité élevée pour "Much Easier" (action 2)
- **Action choisie : "Much Easier"**

**Résultats dans le jeu** :
- Zombies ralentissent de 15%
- Moins de zombies spawnent
- Zombies ont moins de vie
- Joueur devient plus rapide
- Armes font plus de dégâts
- Plus de pickups apparaissent
- **→ Le joueur peut mieux survivre !**

### Scénario : Joueur qui domine

**État observé** :
- Health: 0.9 (90 HP) ← **Trop facile !**
- Damage taken: 0.1 (peu de dégâts) ← **Pas de défi !**
- Zombies alive: 0.2 (peu de zombies) ← **Trop facile !**
- Kill rate: 0.8 (beaucoup de kills) ← **Trop efficace !**

**Prédiction du DDA** :
- Le réseau voit : santé haute + peu de dégâts + peu de zombies
- Il calcule : probabilité élevée pour "Slightly Harder" (action 3)
- **Action choisie : "Slightly Harder"**

**Résultats dans le jeu** :
- Zombies accélèrent de 5%
- Plus de zombies spawnent
- Zombies ont plus de vie
- Joueur devient légèrement plus lent
- Armes font moins de dégâts
- Moins de pickups apparaissent
- **→ Le défi augmente légèrement !**

---

## 6. OBJECTIF DU DDA

Le DDA essaie de maintenir un **"flow zone"** (zone de flux) :

- ✅ **Santé du joueur** : entre 30% et 70%
- ✅ **Nombre de zombies** : entre 3 et 10
- ✅ **Dégâts pris** : modérés (pas trop, pas trop peu)
- ✅ **Taux de kills** : actif (1-5 kills par 5 secondes)

**Si le joueur est en dehors de cette zone** :
- Santé < 30% → DDA choisit "Easier"
- Santé > 70% → DDA choisit "Harder"
- Trop de zombies → DDA choisit "Easier"
- Pas assez de zombies → DDA choisit "Harder"

---

## 7. FRÉQUENCE DES DÉCISIONS

- **Toutes les 5 secondes** (300 frames à 60 FPS)
- Le DDA observe → Prédit → Applique → Attend 5 secondes → Répète

**Pourquoi 5 secondes ?**
- Assez long pour voir des tendances (pas juste un instantané)
- Assez court pour réagir rapidement
- Évite les oscillations (changements trop fréquents)

---

## 8. SMOOTHING (Lissage)

Les ajustements sont **lissés** pour éviter les changements brusques :

- Seulement **30% de l'ajustement** est appliqué à chaque fois
- Exemple : Action "Much Easier" (-15%) → Seulement -4.5% appliqué
- Cela évite que la difficulté "saute" d'un extrême à l'autre

---

## Résumé

1. **Observe** : 9 métriques sur 10 secondes
2. **Prédit** : Action (0-4) avec réseau de neurones
3. **Applique** : Ajuste 5 groupes de paramètres
4. **Résultat** : Jeu plus facile ou plus difficile
5. **Objectif** : Maintenir un défi équilibré (flow zone)

Le DDA apprend de l'expérience (entraînement) pour mieux prédire quelles actions maintiendront le joueur dans la zone de flow !

