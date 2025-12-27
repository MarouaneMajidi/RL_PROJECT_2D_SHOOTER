# RL Zombie Shooter - Reinforcement Learning & Dynamic Difficulty Adjustment

A comprehensive reinforcement learning project featuring a Proximal Policy Optimization (PPO) agent that learns to play a 2D top-down zombie shooter game, plus a Dynamic Difficulty Adjustment (DDA) system that adapts game difficulty in real-time to maintain player engagement.

---

## Table of Contents

- [Overview](#overview)
- [What Makes This Project Interesting](#what-makes-this-project-interesting)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Training & Evaluation](#training--evaluation)
- [What We've Accomplished](#what-weve-accomplished)
- [What's Next](#whats-next)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## Overview

This project tackles an interesting challenge in game AI: can we build an AI that not only learns to play a game, but also learns to adjust the game's difficulty to keep players engaged?

We've implemented two main components:

1. **Player Agent (PPO)**: An AI that learns to survive in a zombie shooter game using reinforcement learning
2. **DDA Agent (PPO)**: A "Game Director" that watches how you play and adjusts difficulty parameters on the fly to keep you in that sweet spot between bored and frustrated

The project includes both traditional state-based RL agents and CNN-based vision agents, along with comprehensive tools for evaluating performance and comparing different approaches.

---

## What Makes This Project Interesting

### Player Agent Features
- Built PPO from scratch using pure PyTorch (no off-the-shelf libraries like Stable Baselines)
- Actor-Critic architecture with separate policy and value networks
- Implements GAE (Generalized Advantage Estimation) for better learning
- Includes both state-based and CNN-based variants
- Carefully designed reward function to encourage good behavior

### Dynamic Difficulty Adjustment
- RL-based DDA agent that learns what difficulty feels right
- Adjusts things like zombie speed, spawn rate, and health in real-time
- Tries to keep players in the "flow zone" where challenge matches skill
- Works with both state observations and visual input
- Comprehensive evaluation framework to measure effectiveness

### Game Implementation
- Classic 2D top-down zombie shooter built with Pygame
- Multiple weapons (Pistol, Machine Gun, Knife) with different characteristics
- Health and ammo pickups scattered around the map
- Traditional wave-based mode for comparison
- DDA mode with adaptive difficulty
- Manual play mode so you can try it yourself

### Evaluation & Testing
- Automated testing that runs hundreds of games and collects statistics
- DDA comparison framework for finding the best approach
- User testing system for blind A/B testing (comparing DDA vs traditional difficulty)
- Detailed metrics collection and visualization tools
- Statistical analysis to understand what's actually working

---

## Project Architecture

Here's how everything fits together:

```
The Game Environment (Pygame-based Zombie Shooter)
                    |
     +--------------+--------------+
     |                             |
Player Agent (PPO)          DDA Agent (PPO)
                            
Learns to:                  Learns to:
- Move strategically        - Watch player performance
- Aim and shoot            - Adjust difficulty parameters
- Survive longer           - Keep challenge balanced
- Switch weapons           - Maintain the "flow zone"

Both agents can use either:
- State-based observations (position, health, etc.)
- CNN-based vision (seeing the game like a human would)
```

---

## Installation

### What You'll Need

- Python 3.11 or newer
- A CUDA-capable GPU is recommended for training (but not required)
- At least 8GB of RAM

### Setting Up

```bash
# Clone the repository
git clone <repository-url>
cd RL_PROJECT

# Install dependencies using uv (recommended)
uv sync

# Or use pip if you prefer
pip install pygame torch numpy matplotlib pandas jupyter

# For GPU support with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Make Sure Everything Works

```bash
python -c "import pygame, torch; print('Installation successful!')"
```

---

## Getting Started

### Try Playing the Game Yourself

```bash
python main.py --mode manual
```

**How to play:**
- Use WASD or Arrow Keys to move around
- Point your mouse to aim
- Left click to shoot zombies
- Press 1, 2, or 3 to switch between weapons
- Press R to restart if things go badly
- Press ESC to quit

### Experience Dynamic Difficulty

```bash
python play_with_dda.py --model checkpoints/dda/backup_best_dda_model.pth
```

The game will adjust its difficulty as you play based on how well you're doing.

### Train Your Own Player Agent

```bash
# Train a state-based agent
python scripts/train_ppo.py

# Train a CNN-based agent that "sees" the game
python scripts_cnn/train_player_cnn.py
```

### Train Your Own DDA Agent

```bash
# State-based DDA
python scripts/train_dda.py

# Vision-based DDA
python scripts_cnn/train_dda_cnn.py
```

### Test How Well an Agent Performs

```bash
# Test the player agent
python scripts/test_agent.py --model checkpoints/player_ppo/best_model.pth

# Test the DDA agent
python test_dda.py
```

### Run a User Testing Session

```bash
python scripts/test_session.py
```

This runs a blind test where you'll play two matches (one with DDA, one without) in random order. The system collects data on your performance and asks for your feedback at the end.

---

## Project Structure

```
RL_PROJECT/
├── agents/                      # RL Agent Implementations
│   ├── ppo_agent/              # State-based player agent
│   └── dda_agent/              # State-based DDA agent
│
├── agents_cnn/                 # Vision-based agents
│   ├── player_cnn/             # CNN player agent
│   └── dda_cnn/                # CNN DDA agent
│
├── scripts/                    # Training and testing scripts
│   ├── train_ppo.py           # Train player agent
│   ├── train_dda.py            # Train DDA agent
│   ├── test_agent.py           # Evaluate player agent
│   └── test_session.py         # Run user testing
│
├── scripts_cnn/                # CNN-specific training
│   ├── train_player_cnn.py
│   └── train_dda_cnn.py
│
├── evaluate_find_best_dda/     # DDA evaluation framework
│   ├── evaluate_dda_agents.py  # Run comprehensive tests
│   ├── analyze_dda_evaluation.ipynb  # Analysis notebook
│   └── data/                   # Evaluation results
│
├── utils/                      # Helper utilities
│   └── metrics_collector.py    # Performance metrics
│
├── checkpoints/                # Saved models
│   ├── player_ppo/            # Trained player agents
│   └── dda/                    # Trained DDA agents
│
├── checkpoints_cnn/            # CNN model checkpoints
│
├── data/                       # Experimental data
│   └── evaluation/            # User testing results
│
├── docs/                       # Detailed documentation
│   ├── DDA_EXPLANATION.md     # How DDA works
│   ├── TRAINING_GUIDE.md      # Training tips
│   └── ...
│
├── assets/                     # Game graphics
│   ├── player_*.gif           # Player sprites
│   ├── zombie*.gif            # Zombie sprites
│   └── background.png         # Background image
│
├── main.py                     # Main game entry point
├── play_with_dda.py           # DDA gameplay
└── test_dda.py                # DDA testing
```

---

## Core Components

### The Player Agent (PPO)

**State-based version:**
- Observes a 98-dimensional state vector including:
  - Player info (position, health, weapon cooldown, facing angle)
  - Up to 15 nearest zombies (position, health, type, distance, angle)
- Can take 6 actions: Move in 4 directions, Shoot, or Stay still
- Gets rewarded for:
  - Killing zombies (+10 points)
  - Staying alive (+0.1 per step)
  - But penalized for taking damage (-5 per 10 HP lost) and dying (-20)

**Vision-based version:**
- Sees the game as an 84x84 RGB image (like looking at a screen)
- Uses a convolutional neural network to understand what's happening
- Same action space and rewards as the state-based version

### The DDA Agent (PPO)

**What it does:**
The DDA agent's job is to keep the game challenging but not frustrating. It watches how you're playing and adjusts difficulty parameters to maintain that "flow state" where you're fully engaged.

**What it observes (9 features, all normalized):**
- Your current health percentage
- How much damage you're taking vs dealing out
- How many zombies you're killing
- Your shooting accuracy
- Number of zombies currently alive
- Time since you last got hit
- How much ammo you have
- Pickups you've grabbed recently

**What it can do (5 actions):**
- Do nothing (difficulty is fine)
- Make it slightly easier
- Make it much easier
- Make it slightly harder
- Make it much harder

**What it adjusts:**
- Zombie characteristics (speed, spawn rate, attack frequency, health)
- Player abilities (movement speed, weapon damage)
- Pickup availability (health and ammo drop rates)
- Weapon cooldowns

**How it learns:**
It gets rewarded for keeping you in the optimal difficulty range (not too easy, not too hard) and penalized when things get too extreme either way.

### The Game Environment

- Arena size: 800x600 pixels
- Two zombie types: Normal (weaker) and Strong (tougher)
- Three weapons: Pistol (balanced), Machine Gun (rapid fire), Knife (melee, unlimited ammo)
- Pickups: Health packs and Machine Gun ammo
- Two modes: Traditional waves (gets harder over time) or DDA (adapts to you)

---

## Training & Evaluation

### Training a Player Agent

```bash
# Basic training for 1 million timesteps
python scripts/train_ppo.py

# Train longer for better performance
python scripts/train_ppo.py --timesteps 2000000

# Customize training parameters
python scripts/train_ppo.py --lr 0.0003 --batch-size 64
```

**What to expect:**
- After 1M timesteps: Survives 400-600 steps, kills 8-15 zombies
- After 2M timesteps: Survives 800-1200 steps, kills 20-35 zombies

### Training a DDA Agent

```bash
# Train state-based DDA
python scripts/train_dda.py

# Train vision-based DDA
python scripts_cnn/train_dda_cnn.py
```

### Evaluating DDA Performance

```bash
# Run comprehensive evaluation (100 games per DDA agent)
cd evaluate_find_best_dda
python evaluate_dda_agents.py

# Analyze the results in a Jupyter notebook
jupyter notebook analyze_dda_evaluation.ipynb
```

**Metrics we measure:**
- How long players survive
- Number of zombies killed
- Percentage of time in the "flow zone" (optimal difficulty)
- What actions the DDA agent takes
- Overall performance scores

---

## What We've Accomplished

### Player Agent Performance
- Successfully learns to play the game through trial and error
- Achieves survival times of 800+ steps
- Kills 20-35+ zombies per game on average
- Shows intelligent behavior like strategic movement, accurate aiming, and weapon switching

### DDA Agent Performance
- Effectively maintains balanced difficulty during gameplay
- Keeps players in the optimal challenge range
- Adapts to different skill levels in real-time
- We've identified the best-performing model: `backup_best_dda_model.pth`

### Evaluation Framework
- Built a comprehensive system for collecting gameplay metrics
- Created tools for statistical analysis and visualization
- Can automatically compare multiple DDA agents
- Successfully identified which DDA configuration works best

### Code Quality
- Clean, modular architecture that's easy to understand and extend
- Thorough documentation throughout
- Reproducible experiments with clear instructions
- Good separation between different components

---

## What's Next

### User Testing & Validation

The next major step is conducting real-world user testing to see if our DDA system actually improves player experience compared to traditional difficulty progression.

**How we'll test it:**
- Blind A/B testing: Each participant plays two matches
  - One match with DDA enabled
  - One match with traditional wave-based difficulty
  - Order is randomized to avoid bias
- We'll collect both objective data (survival time, kills, etc.) and subjective feedback (enjoyment, difficulty perception)
- Players won't know which mode they're playing to avoid bias
- We'll ask them which match they preferred

**Questions we want to answer:**
1. Do players actually prefer DDA over traditional difficulty?
2. Does DDA make the game more engaging and enjoyable?
3. Does DDA maintain a better balance of challenge?
4. What measurable differences exist in player experience?

**How it works:**
- Test session manager: `scripts/test_session.py`
- Uses our best DDA model: `checkpoints/dda/backup_best_dda_model.pth`
- Results automatically saved to: `data/evaluation/`
- Collects detailed metrics and player feedback

**What we hope to learn:**
- Quantitative comparison of DDA vs wave-based difficulty
- Player preference patterns
- Validation that DDA actually works as intended
- Ideas for improving the system further

---

## Documentation

We've written detailed documentation covering different aspects of the project:

- **DDA_EXPLANATION.md** - Deep dive into how the DDA system works
- **TRAINING_GUIDE.md** - Tips and best practices for training agents
- **DDA_DESIGN.md** - Design decisions and architecture explanation
- **DDA_EVALUATION_METRICS.md** - Understanding the evaluation metrics
- **EVALUATION_README.md** - Guide for running user testing

All documentation is in the `docs/` directory.

---

## Contributing

This is an educational and research project, so contributions, suggestions, and feedback are always welcome.

### Ideas for Future Work
- Experiment with different reward functions
- Try alternative RL algorithms (DQN, A3C, etc.)
- More sophisticated state representations
- Add multiplayer support
- Introduce new game mechanics

---

## License

This is an educational project. Feel free to use and modify it for learning purposes.

---

Last Updated: December 2024
