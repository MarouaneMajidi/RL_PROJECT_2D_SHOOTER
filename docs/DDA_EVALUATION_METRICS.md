# Métriques d'Évaluation du DDA

Guide complet des métriques à collecter pour évaluer l'impact du DDA sur l'expérience de jeu.

## Structure du Test

**Deux conditions à comparer :**
1. **Match 1 : Jeu Sans DDA**
   - Difficulté fixe avec progression par vagues
   - Pas d'ajustement dynamique

2. **Match 2 : Jeu Avec DDA**
   - Difficulté ajustée dynamiquement toutes les 5 secondes
   - Même point de départ que Match 1

**Participants :** Chaque personne joue **exactement 2 matches** :
- **Match 1** : Sans DDA OU Avec DDA (ordre randomisé)
- **Match 2** : L'autre condition (Avec DDA OU Sans DDA)
- Chaque match se termine par la mort du joueur

**Collecte de données :**
- **Métriques automatiques** : Collectées automatiquement par le jeu pendant chaque match
- **Questions subjectives** : 3 questions simples posées après chaque match (immédiatement après)

---

## CATÉGORIE 1 : MÉTRIQUES DE SURVIE

### 1.1 Durée de Survie (Survival Time)
**Description :** Temps total de survie avant la mort (en secondes)

**Ce que ça mesure :**
- Capacité du joueur à survivre
- Si le DDA aide à prolonger la survie

**Information donnée :**
- ✅ **Plus long avec DDA** = Le DDA aide à maintenir le joueur en vie
- ❌ **Plus court avec DDA** = Le DDA rend le jeu trop difficile ou déséquilibré
- ≈ **Similaire** = Le DDA n'affecte pas significativement la survie

**Hypothèse :** Le DDA devrait augmenter la durée de survie en maintenant un défi équilibré

---

### 1.2 Score Final (Final Score)
**Description :** Score obtenu à la fin du match (avant la mort)

**Ce que ça mesure :**
- Performance globale du joueur
- Réussite dans le jeu

**Information donnée :**
- ✅ **Score plus élevé avec DDA** = Le DDA améliore la performance
- ❌ **Score plus bas avec DDA** = Le DDA nuit à la performance
- ≈ **Similaire** = Le DDA n'affecte pas le score

**Hypothèse :** Le DDA devrait permettre un score plus élevé (corrélé avec durée de survie)

---

## CATÉGORIE 2 : MÉTRIQUES DE SANTÉ (Flow Zone)

### 2.1 Santé Moyenne (Average Health)
**Description :** Santé moyenne maintenue pendant la session (0-100%)

**Ce que ça mesure :**
- Niveau de pression moyen
- Si le DDA maintient un équilibre

**Information donnée :**
- ✅ **40-60% avec DDA** = Le DDA maintient bien la flow zone (objectif : 30-70%)
- ❌ **< 30% avec DDA** = Le DDA est trop difficile
- ❌ **> 70% avec DDA** = Le DDA est trop facile
- ≈ **Similaire aux deux** = Le DDA n'affecte pas l'équilibre

**Hypothèse :** Le DDA devrait maintenir la santé entre 30-70% (flow zone)

---

### 2.2 Temps dans Flow Zone (Time in Flow Zone)
**Description :** Pourcentage de temps passé avec santé entre 30% et 70%

**Ce que ça mesure :**
- Efficacité du DDA à maintenir l'équilibre
- Qualité de l'expérience de jeu

**Information donnée :**
- ✅ **> 60% avec DDA** = Le DDA maintient bien l'équilibre
- ❌ **< 40% avec DDA** = Le DDA ne maintient pas bien l'équilibre
- ≈ **Similaire** = Le DDA n'améliore pas l'équilibre

**Hypothèse :** Le DDA devrait augmenter le temps dans la flow zone

---

### 2.3 Temps en Danger (< 30% santé) (Time in Danger Zone)
**Description :** Pourcentage de temps passé avec santé < 30%

**Ce que ça mesure :**
- Fréquence des situations critiques
- Si le DDA réduit la frustration

**Information donnée :**
- ✅ **Moins avec DDA** = Le DDA réduit les moments de stress extrême
- ❌ **Plus avec DDA** = Le DDA crée trop de situations difficiles
- ≈ **Similaire** = Le DDA n'affecte pas les moments critiques

**Hypothèse :** Le DDA devrait réduire le temps en danger

---

### 2.4 Temps en Sécurité (> 70% santé) (Time in Safety Zone)
**Description :** Pourcentage de temps passé avec santé > 70%

**Ce que ça mesure :**
- Fréquence des moments trop faciles
- Si le DDA maintient le défi

**Information donnée :**
- ✅ **Moins avec DDA** = Le DDA maintient mieux le défi
- ❌ **Plus avec DDA** = Le DDA rend le jeu trop facile
- ≈ **Similaire** = Le DDA n'affecte pas les moments faciles

**Hypothèse :** Le DDA devrait réduire le temps en sécurité (trop facile)

---

### 2.5 Variance de la Santé (Health Variance)
**Description :** Écart-type de la santé pendant la session

**Ce que ça mesure :**
- Stabilité de l'expérience
- Si le DDA crée des oscillations

**Information donnée :**
- ✅ **Faible variance avec DDA** = Le DDA maintient une expérience stable
- ❌ **Forte variance avec DDA** = Le DDA crée des oscillations (instable)
- ≈ **Similaire** = Le DDA n'affecte pas la stabilité

**Hypothèse :** Le DDA devrait réduire la variance (expérience plus stable)

---

## CATÉGORIE 3 : MÉTRIQUES DE PERFORMANCE

### 3.1 Nombre de Zombies Tués (Total Kills)
**Description :** Nombre total de zombies tués pendant la session

**Ce que ça mesure :**
- Efficacité du joueur
- Si le DDA affecte la capacité à tuer

**Information donnée :**
- ✅ **Plus avec DDA** = Le DDA permet plus d'engagement
- ❌ **Moins avec DDA** = Le DDA rend le combat trop difficile
- ≈ **Similaire** = Le DDA n'affecte pas le combat

**Hypothèse :** Le DDA devrait permettre plus de kills en maintenant un équilibre

---

### 3.2 Taux de Kills (Kill Rate)
**Description :** Nombre de zombies tués par minute

**Ce que ça mesure :**
- Rythme de combat
- Engagement actif

**Information donnée :**
- ✅ **Taux stable avec DDA** = Le DDA maintient un rythme constant
- ❌ **Taux variable avec DDA** = Le DDA crée des variations de rythme
- ≈ **Similaire** = Le DDA n'affecte pas le rythme

**Hypothèse :** Le DDA devrait maintenir un taux de kills stable et actif

---

### 3.3 Précision de Tir (Shooting Accuracy)
**Description :** Pourcentage de tirs qui touchent (hits / shots fired)

**Ce que ça mesure :**
- Compétence du joueur
- Si le DDA affecte la précision

**Information donnée :**
- ✅ **Précision similaire** = Le DDA n'affecte pas la compétence
- ❌ **Précision différente** = Le DDA change les conditions de tir (zombies plus/moins rapides)

**Hypothèse :** La précision devrait être similaire (mesure de compétence, pas de difficulté)

---

### 3.4 Dégâts Infligés (Damage Dealt)
**Description :** Total de dégâts infligés aux zombies

**Ce que ça mesure :**
- Efficacité offensive
- Engagement dans le combat

**Information donnée :**
- ✅ **Plus avec DDA** = Le DDA permet plus d'engagement
- ❌ **Moins avec DDA** = Le DDA rend le combat trop difficile
- ≈ **Similaire** = Le DDA n'affecte pas l'engagement

**Hypothèse :** Le DDA devrait permettre plus de dégâts en maintenant l'équilibre

---

## CATÉGORIE 4 : MÉTRIQUES DE DIFFICULTÉ

### 4.1 Nombre de Zombies Vivants Moyen (Average Zombies Alive)
**Description :** Nombre moyen de zombies vivants simultanément

**Ce que ça mesure :**
- Pression ennemie
- Si le DDA ajuste bien le nombre d'ennemis

**Information donnée :**
- ✅ **3-10 avec DDA** = Le DDA maintient bien la pression (objectif)
- ❌ **< 3 avec DDA** = Le DDA rend trop facile
- ❌ **> 10 avec DDA** = Le DDA crée trop de pression
- ≈ **Similaire** = Le DDA n'affecte pas la pression

**Hypothèse :** Le DDA devrait maintenir 3-10 zombies (zone optimale)

---

### 4.2 Vitesse Moyenne des Zombies (Average Zombie Speed)
**Description :** Vitesse moyenne des zombies pendant la session

**Ce que ça mesure :**
- Ajustement de la difficulté par le DDA
- Pression temporelle

**Information donnée :**
- ✅ **Vitesse adaptée avec DDA** = Le DDA ajuste bien selon la performance
- ❌ **Vitesse constante avec DDA** = Le DDA n'ajuste pas bien
- ≈ **Similaire** = Le DDA n'affecte pas la vitesse

**Hypothèse :** Le DDA devrait ajuster la vitesse selon la performance

---

### 4.3 Taux de Spawn Moyen (Average Spawn Rate)
**Description :** Fréquence moyenne de spawn des zombies (zombies par minute)

**Ce que ça mesure :**
- Rythme de progression
- Ajustement de la pression

**Information donnée :**
- ✅ **Spawn adapté avec DDA** = Le DDA ajuste bien le rythme
- ❌ **Spawn constant avec DDA** = Le DDA n'ajuste pas bien
- ≈ **Similaire** = Le DDA n'affecte pas le spawn

**Hypothèse :** Le DDA devrait ajuster le spawn selon la performance

---

## CATÉGORIE 5 : MÉTRIQUES D'ENGAGEMENT

### 5.1 Durée Totale de Session (Session Duration)
**Description :** Temps total passé à jouer (en minutes)

**Ce que ça mesure :**
- Engagement global
- Si le DDA rend le jeu plus accrocheur

**Information donnée :**
- ✅ **Plus long avec DDA** = Le DDA augmente l'engagement
- ❌ **Plus court avec DDA** = Le DDA réduit l'engagement
- ≈ **Similaire** = Le DDA n'affecte pas l'engagement

**Hypothèse :** Le DDA devrait augmenter la durée de session

---

### 5.2 Taux d'Abandon (Quit Rate)
**Description :** Session terminée par abandon volontaire (vs mort naturelle)

**Ce que ça mesure :**
- Frustration ou ennui pendant la session
- Si le DDA maintient l'intérêt jusqu'à la fin

**Information donnée :**
- ✅ **Moins d'abandons avec DDA** = Le DDA maintient mieux l'intérêt
- ❌ **Plus d'abandons avec DDA** = Le DDA crée frustration/ennui
- ≈ **Similaire** = Le DDA n'affecte pas l'intérêt

**Hypothèse :** Le DDA devrait réduire les abandons pendant la session

---

## CATÉGORIE 6 : MÉTRIQUES DE COMPORTEMENT DDA

### 6.1 Fréquence des Ajustements DDA (DDA Adjustment Frequency)
**Description :** Nombre d'ajustements DDA par minute

**Ce que ça mesure :**
- Activité du DDA
- Si le DDA réagit souvent

**Information donnée :**
- ✅ **Ajustements fréquents** = Le DDA réagit activement
- ❌ **Peu d'ajustements** = Le DDA est trop passif
- ≈ **Stable** = Le DDA maintient bien l'équilibre

**Hypothèse :** Le DDA devrait ajuster régulièrement (toutes les 5 secondes)

---

### 6.2 Distribution des Actions DDA (DDA Action Distribution)
**Description :** Pourcentage de chaque action choisie (Do Nothing, Easier, Harder, etc.)

**Ce que ça mesure :**
- Comportement du DDA
- Si le DDA ajuste dans les deux sens

**Information donnée :**
- ✅ **Distribution équilibrée** = Le DDA ajuste dans les deux sens (bon signe)
- ❌ **Toujours "Easier"** = Le DDA trouve toujours le jeu trop difficile
- ❌ **Toujours "Harder"** = Le DDA trouve toujours le jeu trop facile
- ≈ **Beaucoup "Do Nothing"** = Le DDA maintient bien l'équilibre

**Hypothèse :** Le DDA devrait avoir une distribution équilibrée d'actions

---

### 6.3 Oscillations de Difficulté (Difficulty Oscillations)
**Description :** Nombre de fois que la difficulté change de direction (Easier → Harder → Easier)

**Ce que ça mesure :**
- Stabilité des ajustements
- Si le DDA oscille trop

**Information donnée :**
- ✅ **Peu d'oscillations** = Le DDA est stable
- ❌ **Beaucoup d'oscillations** = Le DDA oscille trop (instable)
- ≈ **Modéré** = Le DDA ajuste de manière stable

**Hypothèse :** Le DDA devrait avoir peu d'oscillations (ajustements lisses)

---

## CATÉGORIE 7 : QUESTIONS SUBJECTIVES (3 Questions Simples)

**Important :** Ces 3 questions sont posées **immédiatement après chaque match**.

### 7.1 Plaisir (Enjoyment)
**Question :** "À quel point avez-vous apprécié ce match ?"
**Échelle :** 1 (Pas du tout) à 5 (Beaucoup)

**Ce que ça mesure :**
- Plaisir subjectif global
- Si le DDA améliore l'expérience

**Information donnée :**
- ✅ **Score plus élevé avec DDA** = Le DDA améliore le plaisir
- ❌ **Score plus bas avec DDA** = Le DDA réduit le plaisir
- ≈ **Similaire** = Le DDA n'affecte pas le plaisir

**Hypothèse :** Le DDA devrait augmenter le score de plaisir (+0.5 à +1.0 point)

---

### 7.2 Difficulté (Difficulty)
**Question :** "Le jeu était-il trop facile, trop difficile, ou équilibré ?"
**Échelle :** 1 (Trop facile) à 5 (Trop difficile), 3 = Équilibré

**Ce que ça mesure :**
- Perception de l'équilibre
- Si le DDA crée un bon équilibre perçu

**Information donnée :**
- ✅ **Score ≈ 3 avec DDA** = Le DDA maintient bien l'équilibre
- ❌ **Score extrême (1 ou 5) avec DDA** = Le DDA crée déséquilibre
- ≈ **Similaire** = Le DDA n'affecte pas la perception

**Hypothèse :** Le DDA devrait maintenir un score proche de 3 (équilibré)

---

### 7.3 Préférence (Préférence Globale)
**Question :** "Quelle version avez-vous préférée ?"
**Options :** 
- Le premier match (Sans DDA / Avec DDA selon l'ordre)
- Le deuxième match (Avec DDA / Sans DDA selon l'ordre)
- Aucune préférence (Les deux étaient similaires)

**Ce que ça mesure :**
- **Métrique clé subjective** : Préférence directe entre les deux versions
- Impact global du DDA sur l'expérience

**Information donnée :**
- ✅ **Majorité préfère le match avec DDA** = Le DDA améliore significativement l'expérience
- ❌ **Majorité préfère le match sans DDA** = Le DDA nuit à l'expérience
- ≈ **Aucune préférence majoritaire** = Le DDA n'a pas d'impact notable

**Note :** Cette question est posée **après les deux matches** (à la toute fin)

**Hypothèse :** Plus de 60% devraient préférer le match avec DDA

---

## CATÉGORIE 8 : MÉTRIQUES COMPARATIVES

### 8.1 Ratio de Performance (Performance Ratio)
**Description :** (Métrique avec DDA) / (Métrique sans DDA)

**Ce que ça mesure :**
- Amélioration relative
- Impact du DDA

**Information donnée :**
- ✅ **Ratio > 1.0** = Le DDA améliore cette métrique
- ❌ **Ratio < 1.0** = Le DDA réduit cette métrique
- ≈ **Ratio ≈ 1.0** = Le DDA n'affecte pas cette métrique

**Exemple :** Ratio de survie = 1.2 signifie 20% plus de survie avec DDA

---

### 8.2 Amélioration Relative (Relative Improvement)
**Description :** ((Métrique avec DDA) - (Métrique sans DDA)) / (Métrique sans DDA) × 100%

**Ce que ça mesure :**
- Pourcentage d'amélioration
- Impact en termes relatifs

**Information donnée :**
- ✅ **+20%** = Le DDA améliore de 20%
- ❌ **-10%** = Le DDA réduit de 10%
- ≈ **0%** = Pas d'effet

---

## MÉTRIQUES PRIORITAIRES (Top 10)

Pour une évaluation rapide, concentrez-vous sur ces 10 métriques :

### Métriques Automatiques (Top 6) :
1. **Durée de Survie** - Impact direct sur l'expérience
2. **Temps dans Flow Zone** - Objectif principal du DDA
3. **Santé Moyenne** - Équilibre général
4. **Temps en Danger (< 30%)** - Réduction de frustration
5. **Nombre de Zombies Tués** - Performance offensive
6. **Variance de la Santé** - Stabilité

### Questions Subjectives (Top 3) :
7. **Plaisir** - "À quel point avez-vous apprécié ce match ?" (1-5)
8. **Difficulté** - "Le jeu était-il équilibré ?" (1-5, 3=équilibré)
9. **Préférence** - "Quelle version avez-vous préférée ?" (après les deux matches)

---

## INTERPRÉTATION DES RÉSULTATS

### Signes que le DDA fonctionne bien :

✅ **Métriques objectives :**
- Durée de survie : +15-30%
- Temps dans flow zone : +20-40%
- Temps en danger : -30-50%
- Variance de santé : -20-30%

✅ **Métriques subjectives :**
- Plaisir : +0.5-1.0 point (sur 5)
- Difficulté : Score équilibré (≈3/5)
- Préférence : Majorité préfère le match avec DDA (> 60%)

### Signes que le DDA ne fonctionne pas bien :

❌ **Métriques objectives :**
- Durée de survie : -10% ou plus
- Temps dans flow zone : Pas d'amélioration
- Temps en danger : Augmentation
- Variance de santé : Augmentation (oscillations)

❌ **Métriques subjectives :**
- Plaisir : Pas d'amélioration ou réduction
- Difficulté : Score extrême (1 ou 5)
- Préférence : Majorité préfère le match sans DDA

---

## RECOMMANDATIONS DE COLLECTE

### Structure de collecte :

**Pour chaque joueur :**
1. **Match 1** : Jouer jusqu'à la mort (Sans DDA OU Avec DDA - ordre randomisé)
2. **Questions après Match 1** : Poser les 2 premières questions (7.1 Plaisir, 7.2 Difficulté)
3. **Match 2** : Jouer jusqu'à la mort (L'autre condition)
4. **Questions après Match 2** : Poser les 2 premières questions (7.1 Plaisir, 7.2 Difficulté)
5. **Question finale** : Préférence (7.3) - "Quelle version avez-vous préférée ?"

### Métriques automatiques (collectées par le jeu) :

**Enregistrer toutes les 1 seconde pendant la session :**
- Santé du joueur
- Nombre de zombies vivants
- Nombre de zombies tués (cumulatif)
- Dégâts pris ce frame
- Dégâts infligés ce frame
- Tirs tirés/touchés
- Pickups collectés
- Paramètres de difficulté actuels (si DDA activé)
- Action DDA choisie (si DDA activé)

**Calculer après la session :**
- Toutes les métriques des Catégories 1-6
- Statistiques (moyennes, variances, pourcentages)

### Questions subjectives (posées manuellement) :

**Après chaque match, poser (2 questions) :**
1. **Plaisir** : "À quel point avez-vous apprécié ce match ?" (1-5)
2. **Difficulté** : "Le jeu était-il trop facile, trop difficile, ou équilibré ?" (1-5, 3=équilibré)

**Après les deux matches, poser (1 question) :**
3. **Préférence** : "Quelle version avez-vous préférée ?" (Premier match / Deuxième match / Aucune préférence)

### Métadonnées à enregistrer :

**Pour chaque joueur :**
- ID du joueur
- Ordre de jeu (DDA en premier ou en second)
- Date et heure de chaque session

**Pour chaque match :**
- Condition (Avec DDA / Sans DDA)
- Durée du match (jusqu'à la mort)
- Ordre (Match 1 ou Match 2)

---

## ANALYSES STATISTIQUES RECOMMANDÉES

### Pour les métriques automatiques (quantitatives) :

1. **Test t de Student** (paired) : Comparer avec/sans DDA pour chaque métrique
   - Chaque joueur a deux valeurs (sans DDA, avec DDA)
   - Test si la différence est significative

2. **Effect Size (Cohen's d)** : Mesurer l'ampleur de l'effet
   - d > 0.5 = Effet moyen
   - d > 0.8 = Effet large

3. **Intervalle de confiance à 95%** : Précision des résultats
   - Montrer la variabilité entre joueurs

4. **Visualisations** : 
   - Box plots : Comparer distributions avec/sans DDA
   - Histogrammes : Distributions de chaque métrique
   - Graphiques temporels : Évolution de la santé au fil du temps

### Pour les questions subjectives (qualitatives) :

1. **Test de Wilcoxon** (non-paramétrique, paired) : Comparer scores avec/sans DDA
   - Alternative au test t pour données ordinales (échelles 1-5)

2. **Test du Chi-carré** : Pour la question de préférence
   - Comparer proportions (Sans DDA / Avec DDA / Aucune préférence)

3. **Visualisations** :
   - Bar charts : Comparer scores moyens
   - Pie charts : Distribution des préférences

---

## RÉSUMÉ DE LA COLLECTE

### Ce qui est collecté automatiquement (par le jeu) :

**Pendant chaque session :**
- Santé du joueur (chaque seconde)
- Nombre de zombies (chaque seconde)
- Événements (kills, dégâts, pickups)
- Paramètres DDA (si activé)

**Calculé après chaque session :**
- Toutes les métriques des Catégories 1-6 (Survie, Santé, Performance, Difficulté, Engagement, Comportement DDA)

### Ce qui est collecté manuellement (questions) :

**Après chaque session :**
- 6 questions sur échelle 1-5 (Enjoyment, Difficulty, Frustration, Flow, Control, Engagement)

**Après les deux sessions :**
- 1 question de préférence (Sans DDA / Avec DDA / Aucune préférence)

### Format de données recommandé :

**Fichier CSV par joueur :**
```
player_id, session_number, condition, order, date, duration, survival_time, ...
```

**Fichier CSV des questions :**
```
player_id, session_number, condition, enjoyment, perceived_difficulty, frustration, flow_state, control, engagement, preference
```

---

## RÉSUMÉ DE LA STRUCTURE DU TEST

### Pour chaque joueur :

1. **Match 1** : Jouer jusqu'à la mort (Sans DDA OU Avec DDA - ordre randomisé)
   - Métriques automatiques collectées pendant le match
   - 2 questions après le match (Plaisir, Difficulté)

2. **Match 2** : Jouer jusqu'à la mort (L'autre condition)
   - Métriques automatiques collectées pendant le match
   - 2 questions après le match (Plaisir, Difficulté)

3. **Question finale** : Préférence (après les deux matches)
   - "Quelle version avez-vous préférée ?"

### Total : 2 matches par joueur, 3 questions simples

---

## CONCLUSION

Ces métriques vous permettront d'évaluer objectivement et subjectivement si le DDA améliore l'expérience de jeu. 

**Rappel important :**
- Chaque joueur joue **exactement 2 matches** (un sans DDA, un avec DDA)
- Chaque match se termine par la mort du joueur
- 3 questions simples seulement (2 après chaque match + 1 préférence finale)
- Toutes les métriques sont comparées entre les deux matches pour chaque joueur

**Concentrez-vous sur les 9 métriques prioritaires** pour une évaluation rapide et efficace.

