# DDA Testing Guide

Guide pour tester l'agent DDA dans deux modes différents.

## Modes de Test

### 1. Mode Manuel (Vous jouez)

Dans ce mode, **vous contrôlez le joueur** et l'agent DDA ajuste la difficulté en fonction de votre performance.

#### Lancer le mode manuel:

```bash
python play_with_dda.py
```

Ou avec un modèle spécifique:

```bash
python play_with_dda.py --model checkpoints/dda/best_dda_model.pth
```

#### Contrôles:
- **WASD** ou **Flèches** - Se déplacer
- **Souris** - Viser
- **Clic gauche** - Tirer
- **1/2/3** - Changer d'arme (Pistol/Machinegun/Knife)
- **R** - Recommencer
- **ESC** - Quitter

#### Affichage DDA:
En haut à droite de l'écran, vous verrez:
- **DDA**: Action choisie par l'agent (Do Nothing, Easier, Harder, etc.)
- **Z.Speed**: Vitesse actuelle des zombies
- **Spawn**: Taux de spawn actuel

L'agent DDA ajuste la difficulté **toutes les 5 secondes** en fonction de:
- Votre santé
- Les dégâts pris
- Le nombre de zombies
- Votre taux de kills

---

### 2. Mode Agent (PPO Agent joue)

Dans ce mode, l'**agent PPO entraîné** joue et l'agent DDA ajuste la difficulté.

#### Lancer le mode agent:

```bash
python test_dda.py --mode agent
```

Ou avec des modèles spécifiques:

```bash
python test_dda.py --mode agent --dda-model checkpoints/dda/best_dda_model.pth --player-model checkpoints/best_model.pth
```

#### Ce que vous verrez:
- Le jeu s'affiche avec l'agent PPO qui joue
- Toutes les 5 secondes, dans la console:
  - Action DDA choisie
  - État du joueur (santé, zombies)
  - Paramètres de difficulté ajustés

#### Contrôles:
- **ESC** - Quitter
- **R** - Recommencer l'épisode

---

## Script Unifié

Vous pouvez aussi utiliser le script unifié qui choisit le mode:

```bash
# Mode manuel
python test_dda.py --mode manual

# Mode agent
python test_dda.py --mode agent
```

---

## Interprétation des Résultats

### Actions DDA:
- **Do Nothing** → Difficulté équilibrée, pas besoin d'ajustement
- **Slightly Easier** → Le joueur a un peu de mal, réduire légèrement
- **Much Easier** → Le joueur est en difficulté, réduire beaucoup
- **Slightly Harder** → Le joueur gère bien, augmenter légèrement
- **Much Harder** → Le joueur domine, augmenter beaucoup

### Signes d'un bon DDA:
✅ Le joueur survit plus longtemps
✅ La santé reste entre 30-70% (zone de flow)
✅ Le nombre de zombies reste entre 3-10
✅ Les actions DDA alternent entre "easier" et "harder" (pas toujours la même)

### Signes d'un DDA qui apprend:
⚠️ Au début: beaucoup de morts rapides, actions "Much Easier"
✅ Après entraînement: survie plus longue, actions plus équilibrées

---

## Dépannage

### Erreur: "DDA model not found"
```bash
# Entraîner d'abord l'agent DDA
python train_dda.py --player-model checkpoints/best_model.pth
```

### Erreur: "Player model not found" (mode agent)
```bash
# Entraîner d'abord l'agent joueur
python train_ppo.py
```

### Le jeu ne s'affiche pas
- Vérifiez que vous avez pygame installé
- Vérifiez que les assets sont dans le dossier `assets/`
- Essayez le mode headless pour le training: `python train_dda.py`

---

## Exemples de Test

### Test 1: Vérifier que le DDA réagit
1. Lancez le mode manuel
2. Laissez-vous mourir rapidement
3. Observez: l'agent devrait choisir "Much Easier" ou "Slightly Easier"

### Test 2: Vérifier l'équilibrage
1. Lancez le mode agent
2. Observez les actions DDA dans la console
3. Si le joueur survit longtemps avec santé > 70%, l'agent devrait choisir "Harder"
4. Si le joueur meurt vite, l'agent devrait choisir "Easier"

### Test 3: Comparer avec/sans DDA
1. Jouez sans DDA (mode normal)
2. Jouez avec DDA (mode manuel)
3. Comparez: avec DDA, la difficulté devrait s'adapter à votre niveau

---

## Notes

- L'agent DDA agit toutes les **5 secondes** (300 frames à 60 FPS)
- Les ajustements sont **lissés** pour éviter les oscillations
- Les paramètres sont **bornés** pour rester dans des limites raisonnables
- Le DDA n'affecte **pas** les contrôles du joueur, seulement la difficulté

