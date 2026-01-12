# RL Zombie Shooter - Reinforcement Learning & Dynamic Difficulty Adjustment

A comprehensive reinforcement learning project featuring a Proximal Policy Optimization (PPO) agent that learns to play a 2D top-down zombie shooter game, plus a Dynamic Difficulty Adjustment (DDA) system that adapts game difficulty in real-time to maintain player engagement.

---

## Table of Contents

- [Overview](#overview)
- [Demos](#demo)
- [What Makes This Project Interesting](#what-makes-this-project-interesting)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Training & Evaluation](#training--evaluation)
- [Training and Comparison Pipeline](#training-and-comparison-pipeline)
- [What's Next](#whats-next)
- [Contributing](#contributing)

---

## Overview

This project tackles an interesting challenge in game AI: can we build an AI that learns to adjust the game's difficulty to keep players engaged?

We've implemented two main components:

1. **Player Agent (PPO)**: An AI that learns to survive in a zombie shooter game using reinforcement learning

2. **DDA Agent (PPO)**: A "Game Director" that watches how you play and adjusts difficulty parameters on the fly to keep you in that sweet spot between bored and frustrated

The project includes both traditional state-based RL agents and CNN-based vision agents, along with comprehensive tools for evaluating performance and comparing different approaches.

---

## Demos

### Feature-Based PPO Player Agent
The agent observes a handcrafted state vector and learns movement, aiming, and shooting strategies using PPO.
![Feature-Based PPO Player Agent](demos/feature_based_agent_playing.gif)

---

### Vision-Based PPO Player Agent (CNN)
The agent learns directly from stacked RGB frames, perceiving the game world visually like a human player.
![Vision-Based PPO Player Agent](demos/vision-based-player.gif)

---

### Dynamic Difficulty Adjustment (DDA) Agent
A reinforcement learning–based game director that adjusts game difficulty every 5 seconds based on player performance to maintain engagement.
![Dynamic Difficulty Adjustment Agent](demos/dda_agent_with_human_player.gif)

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

- Works with both state observations and visual input

- Comprehensive evaluation framework to measure effectiveness

### Game Implementation

- Classic 2D top-down zombie shooter built with Pygame

- Multiple weapons (Pistol, Machine Gun) with different characteristics

- Health and ammo pickups scattered around the map

- Traditional wave-based mode for comparison

- DDA mode with adaptive difficulty

- Manual play mode so you can try it yourself

![Image](https://github.com/user-attachments/assets/b6c58d0d-64c2-4fe9-8c06-b4bcf7c9f9ff)

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

- Switch weapons           

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

- Sees the game as 4 consecutive 128×96 RGB frames (stacked over time), similar to observing motion on a screen.

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

- Three weapons: Pistol (balanced) and Machine Gun (rapid fire)

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

- Percentage of time in the "target zone"

- What actions the DDA agent takes

- Overall performance scores

---

## Training and Comparison Pipeline

### Overview

Our pipeline consists of two main phases: first training and selecting the best player agent, then using that agent to train and evaluate multiple DDA agents. The process involves systematic evaluation, statistical analysis, and careful selection criteria.

### Phase 1: Training and Selecting the Best Player Agent

**Step 1.1: Training Multiple Player Agents**

We started by training multiple player agents using the PPO algorithm. Each agent learned to play the game through reinforcement learning, developing strategies for movement, aiming, shooting, and survival.

**Step 1.2: Evaluating Player Agents**

To find the best player agent, we used an automated testing script (`find_best_manual_model/test_manual_models.py`) that:

1. Loads all trained player models from checkpoints
2. Runs 100 games per model
3. Collects metrics: survival time, zombies killed, wave achieved
4. Saves results to CSV file

**Step 1.3: Statistical Analysis and Selection**

Using a Jupyter notebook (`find_best_manual_model/best_model_overall.ipynb`), we analyzed the results:

1. **Descriptive Statistics**: Calculated mean, median, standard deviation, min, max, quartiles for each model
2. **Statistical Tests**: Performed Kruskal-Wallis test to check for significant differences
3. **Selection Criteria**: Chose the model with:
   - Higher median survival time
   - Lower variability (standard deviation and IQR)
   - More consistent performance

**Result: best_model_marouane selected**

- Mean survival: 3,298 steps (vs 2,902 for alternative)
- Median survival: 3,417 steps
- Lower variability: 1,171 std (vs 1,571 for alternative)
- More consistent performance across 100 games

**Player Agent Selection Pipeline:**

```
Train Multiple Player Agents
    │
    ├── best_model_marouane
    ├── best_model_sohaib
    └── (other variants Vision based)
    │
    ▼
Test Each Agent (100 games)
    │
    ├── Collect: survival_time, zombies_killed, wave_achieved
    └── Save to CSV
    │
    ▼
Statistical Analysis (Jupyter Notebook)
    │
    ├── Descriptive statistics
    ├── Kruskal-Wallis test
    └── Compare variability
    │
    ▼
Select Best Player Agent
    │
    └── best_model_marouane (highest median, lowest variability)
```

### Phase 2: Training and Evaluating DDA Agents

**Step 2.1: Training Multiple DDA Agents**

Once we had the best player agent (best_model_marouane), we used it to train multiple DDA agents. The DDA agents learned to adjust game difficulty by observing how the player agent performed.

We trained several DDA agents with different configurations:

**State-based DDA Agents:**
- `RL_best_dda_model_1`: Trained with specific hyperparameters
- `RL_best_dda_model_2`: Alternative configuration
- `RL_backup_best_dda_model`: Backup model with different training schedule

**CNN-based DDA Agents:**
- `CNN_best_model`: Vision-based DDA agent using convolutional networks

**Step 2.2: Automated Evaluation**

We developed an automated evaluation pipeline (`evaluate_find_best_dda/evaluate_dda_agents.py`) that:

1. Loads all trained DDA models from checkpoints directories
2. Runs 100 games per DDA agent using the fixed best player agent (best_model_marouane.pth)
3. Collects comprehensive metrics for each game:
   - Survival time (steps and seconds)
   - Total zombies killed
   - Average health percentage
   - Time spent in flow zone (optimal difficulty range)
   - Time spent in danger zone (too difficult)
   - Time spent in safety zone (too easy)
   - Health stability (standard deviation)
   - Near-death count (times health dropped below 20%)
   - Near-death time percent (time spent with health < 20%)
   - Action distribution (counts of each DDA action)
   - Action oscillations

4. Saves results to CSV files for analysis:
   - `dda_evaluation_agents.csv`: Game-level metrics
   - `dda_evaluation_intervals.csv`: Interval-level metrics (5-second intervals)

**Step 2.3: The Challenge of Selecting the Best DDA Agent**

Selecting the best DDA agent is more complex than selecting the best player agent. Here's why:

**The Problem:**
If we simply choose the DDA agent that makes players survive the longest, we might select an agent that always makes the game easier. This would result in:
- High survival times (players never die)
- But poor gameplay experience (too easy, boring)
- No challenge or engagement

**The Solution:**
We need a DDA agent that maintains balanced difficulty - challenging enough to be engaging, but not so hard that players die quickly. We use our intuition and multiple criteria:

1. **Action Distribution**: We examine what actions the DDA agent takes
   - A good DDA should use a balanced mix of actions
   - If it always chooses "much easier", that's a red flag
   - If it oscillates too much (harder, easier, harder, easier), that's also problematic

2. **Time in Critical Health (<20%)**: We measure how much time players spend with health below 20%
   - Too much time in critical health means the DDA isn't helping when needed
   - Too little time might mean the game is too easy
   - A good DDA should minimize critical health time while maintaining challenge

3. **Flow Zone Time**: Percentage of time in optimal difficulty range
   - Higher is better, but not at the expense of making the game too easy

**DDA Agent Selection Pipeline:**

```
Train Multiple DDA Agents
    │
    ├── RL_best_dda_model_1
    ├── RL_best_dda_model_2
    ├── RL_backup_best_dda_model
    └── CNN_best_model
    │
    ▼
Evaluate Each DDA (100 games with best player agent)
    │
    ├── Collect: survival, kills, flow zone, danger zone
    ├── Collect: action distribution
    ├── Collect: time in critical health (<20%)
    └── Save to CSV
    │
    ▼
Statistical Analysis (Jupyter Notebook)
    │
    ├── Descriptive statistics
    ├── Action distribution analysis
    ├── Time in critical health analysis
    ├── Composite scoring
    └── Visualizations
    │
    ▼
Selection Based on Intuition & Multiple Criteria
    │
    ├── NOT just survival time (would favor "always easier")
    ├── Action distribution (balanced actions)
    ├── Time in critical health <20% (minimize but maintain challenge)
    ├── Flow zone percentage
    └── Composite score
    │
    ▼
Selected: backup_best_dda_model.pth
    (Best balance of challenge and engagement)
```

**Step 2.4: Statistical Comparison**

Using a Jupyter notebook (`analyze_dda_evaluation.ipynb`), we performed detailed analysis:

1. **Descriptive Statistics**: Mean, standard deviation, min, max for each metric per agent

2. **Action Distribution Analysis**: Examined the frequency of each action type:
   - Do nothing
   - Slightly easier / Much easier
   - Slightly harder / Much harder
   - Identified agents with unbalanced or oscillating behavior

3. **Critical Health Analysis**: Analyzed time spent with health < 20%:
   - Lower is generally better (DDA is helping)
   - But must be balanced with maintaining challenge

4. **Ranking System**: Ranked agents by multiple criteria to find balanced performance

5. **Visualization**: Generated comparison plots:
   - Survival time distributions
   - Flow zone comparisons
   - Action distribution charts
   - Critical health time analysis
   - Correlation matrices

**Step 2.5: Best Model Selection**

Based on the comprehensive analysis considering action distribution, critical health time, and composite scores, we selected:

**Best DDA Agent: backup_best_dda_model.pth**

**Selection Rationale:**
- Balanced action distribution (not always easier)
- Low time in critical health (<20%) while maintaining challenge
- Best overall balance between challenge and player assistance

**Performance Results:**
- Survival Time: 90.2 seconds (mean), 45.83 seconds (std)
- Total Kills: 30.62 (mean), 14.98 (std)
- Flow Zone Time: 8.53% (mean)
- Danger Zone Time: 8.06% (mean)
- Near-death Count: 1.04 (mean) - low critical health events
- Balanced action distribution

### Complete Pipeline Summary

```
PHASE 1: Player Agent Selection
    Train Player Agents → Test (100 games) → Statistical Analysis → Select Best
    Result: best_model_marouane

PHASE 2: DDA Agent Selection
    Train DDA Agents (using best player) → Evaluate (100 games each) 
    → Analyze Action Distribution & Critical Health → Select Best
    Result: backup_best_dda_model.pth
```

This two-phase approach ensures we first have a strong player agent, then use it to train and evaluate DDA agents with careful consideration of multiple factors beyond simple survival time.

---

## What's Next

### User Testing & Validation

The next major step is conducting real-world user testing to see if our DDA system actually improves player experience compared to traditional difficulty progression.

**How we'll test it:**

- Blind A/B testing: Each participant plays two matches

  - One match with DDA enabled (using backup_best_dda_model.pth)

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
