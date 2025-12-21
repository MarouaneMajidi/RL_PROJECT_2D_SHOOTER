# Dynamic Difficulty Adjustment (DDA) System

A reinforcement learning agent that acts as a "Game Director" to dynamically adjust game difficulty based on player performance, maintaining a balanced challenge.

## Overview

The DDA system uses a PPO-based RL agent that observes game state and adjusts difficulty parameters at fixed intervals to keep the player in a "flow zone" - neither too easy nor too hard.

## Architecture

### Components

1. **State Extractor** (`state_extractor.py`)
   - Extracts normalized state features from game metrics
   - Uses sliding windows (10 seconds) for short-term aggregated metrics
   - State features (all normalized to [0, 1]):
     - `player_health_ratio`
     - `average_damage_taken_last_10s`
     - `average_damage_dealt_last_10s`
     - `kill_rate_last_10s`
     - `shooting_accuracy_last_10s`
     - `number_of_zombies_alive`
     - `time_since_last_player_hit`
     - `ammo_ratio`
     - `pickups_collected_last_10s`

2. **Difficulty Manager** (`difficulty_manager.py`)
   - Manages difficulty parameters grouped into categories:
     - **Enemy Pressure**: ZOMBIE_SPEED, SPAWN_RATE, ZOMBIE_ATTACK_COOLDOWN
     - **Enemy Durability**: ZOMBIE_HEALTH
     - **Player Power**: PLAYER_SPEED, WEAPON_DAMAGE
     - **Resource Generosity**: HEALTH_PICKUP_DROP_PROBABILITY, MACHINEGUN_PICKUP_DROP_PROBABILITY
     - **Combat Tempo**: WEAPON_COOLDOWNS
   - Applies smooth, bounded adjustments based on DDA actions
   - Uses smoothing to prevent oscillations

3. **DDA Environment** (`dda_env.py`)
   - Wraps the game environment to provide DDA agent's perspective
   - Tracks metrics, extracts state, applies difficulty adjustments
   - Computes rewards for balanced challenge
   - Acts at fixed intervals (default: every 5 seconds)

4. **DDA Agent** (`dda_agent.py`)
   - PPO-based RL agent
   - Discrete action space (5 actions):
     - 0: Do nothing
     - 1: Slightly easier
     - 2: Much easier
     - 3: Slightly harder
     - 4: Much harder

5. **Model** (`model.py`)
   - Simple actor-critic neural network
   - Shared feature extractor with separate actor and critic heads

## Reward Function

The reward function encourages balanced challenge:

- **Survival**: +1 if player alive, -5 if player dies
- **Health Balance**: Bonus if health is in target zone (0.3-0.7), penalty if outside
- **Zombie Count Balance**: Bonus if 3-10 zombies alive, penalty if too few/many
- **Damage Taken**: Penalty if damage taken is too high
- **Kill Rate**: Bonus for active engagement (1-5 kills per 5 seconds)

## Training

### Prerequisites

1. A trained player agent model (default: `checkpoints/best_model.pth`)
2. The game environment set up

### Training Command

```bash
python train_dda.py --player-model checkpoints/best_model.pth
```

### Options

- `--player-model`: Path to trained player agent model
- `--checkpoint`: Path to DDA checkpoint to resume from
- `--render`: Render during training (slower)

### Training Process

1. Loads the trained player agent
2. Creates DDA environment wrapping the game
3. Trains DDA agent using PPO
4. DDA agent adjusts difficulty every 5 seconds (300 frames at 60 FPS)
5. Saves best model and periodic checkpoints

## Usage

After training, the DDA agent can be used to dynamically adjust difficulty during gameplay:

```python
from dda_agent import DDAAgent, DDAConfig, DDAEnvironment
from ppo_agent import PPOAgent, PPOConfig, ZombieShooterEnv

# Load DDA agent
dda_config = DDAConfig()
dda_agent = DDAAgent(dda_config)
dda_agent.load("checkpoints/dda/best_dda_model.pth")

# Create environment
ppo_config = PPOConfig()
game_env = ZombieShooterEnv(ppo_config)
dda_env = DDAEnvironment(game_env, base_params, action_interval=300)

# Run with DDA
state = dda_env.reset()
while True:
    action = dda_agent.select_action(state, deterministic=True)[0]
    state, reward, done, info = dda_env.step(action)
    if done:
        break
```

## Design Choices

1. **State Space**: Uses short-term aggregated metrics (sliding windows) rather than full history to maintain Markovian property
2. **Action Space**: Discrete actions for simplicity and interpretability
3. **Parameter Grouping**: Groups related parameters to reduce action space complexity
4. **Smoothing**: Limits parameter changes per step to prevent oscillations
5. **Fixed Intervals**: DDA acts every 5 seconds, not every frame, for stability
6. **Reward Design**: Focuses on maintaining balanced challenge (flow zone) rather than maximizing survival or kills

## Files

- `__init__.py`: Package initialization
- `state_extractor.py`: State extraction with sliding windows
- `difficulty_manager.py`: Difficulty parameter management
- `dda_env.py`: DDA environment wrapper
- `dda_agent.py`: DDA agent implementation (PPO)
- `model.py`: Neural network model
- `config.py`: Configuration
- `memory.py`: Rollout buffer for PPO
- `README.md`: This file

## Notes

- The DDA agent does NOT modify player controls or gameplay logic
- It only adjusts difficulty parameters at fixed intervals
- Training uses the existing trained player agent
- The system is designed to maintain a balanced challenge, not to make the game easier or harder overall

