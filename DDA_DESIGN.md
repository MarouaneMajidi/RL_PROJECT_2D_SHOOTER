# Dynamic Difficulty Adjustment (DDA) System - Design Document

## Overview

This document describes the design and implementation of a Dynamic Difficulty Adjustment (DDA) system for the 2D top-down shooter game. The DDA system uses a reinforcement learning agent that acts as a "Game Director" to dynamically adjust game difficulty parameters based on player performance, maintaining a balanced challenge.

## Key Design Principles

1. **Separation of Concerns**: The DDA agent does NOT control player actions or modify gameplay mechanics. It only adjusts difficulty parameters.

2. **Markovian State**: The state space uses short-term aggregated metrics (sliding windows) rather than full history to maintain the Markov property.

3. **Grouped Parameters**: Difficulty parameters are grouped into logical categories to reduce action space complexity.

4. **Smooth Adjustments**: Parameter changes are smoothed to prevent oscillations.

5. **Fixed Intervals**: The DDA agent acts at fixed intervals (every 5 seconds) rather than every frame for stability.

## Architecture

### 1. State Space (9 dimensions, all normalized to [0, 1])

The state represents the current game situation using short-term aggregated metrics:

- `player_health_ratio`: Current player health / 100
- `average_damage_taken_last_10s`: Average damage taken per second (normalized)
- `average_damage_dealt_last_10s`: Average damage dealt per second (normalized)
- `kill_rate_last_10s`: Kills per second (normalized)
- `shooting_accuracy_last_10s`: Hit rate (hits / shots fired)
- `number_of_zombies_alive`: Current zombie count (normalized by max)
- `time_since_last_player_hit`: Time since last hit (normalized)
- `ammo_ratio`: Current ammo / max ammo
- `pickups_collected_last_10s`: Pickups per second (normalized)

**Why these features?**
- They capture the current challenge level without requiring full history
- They are approximately Markovian (current state + short history)
- They directly relate to difficulty balance (health, pressure, engagement)

### 2. Action Space (5 discrete actions)

- **0**: Do nothing (maintain current difficulty)
- **1**: Slightly easier (-5% adjustment)
- **2**: Much easier (-15% adjustment)
- **3**: Slightly harder (+5% adjustment)
- **4**: Much harder (+15% adjustment)

**Why discrete actions?**
- Simpler to interpret and debug
- Reduces action space complexity
- High-level control matches the task (difficulty adjustment)

### 3. Difficulty Parameter Groups

Parameters are grouped into logical categories:

#### Enemy Pressure
- `ZOMBIE_SPEED`: How fast zombies move
- `SPAWN_RATE`: How often zombies spawn (lower = faster)
- `ZOMBIE_ATTACK_COOLDOWN`: Time between zombie attacks

#### Enemy Durability
- `ZOMBIE_NORMAL_HEALTH`: Health of normal zombies
- `ZOMBIE_STRONG_HEALTH`: Health of strong zombies

#### Player Power
- `PLAYER_SPEED`: How fast the player moves
- `PISTOL_DAMAGE`: Damage per pistol shot
- `MACHINEGUN_DAMAGE`: Damage per machinegun shot

#### Resource Generosity
- `HEALTH_PICKUP_DROP_PROBABILITY`: Chance of health pickup drop
- `MACHINEGUN_PICKUP_DROP_PROBABILITY`: Chance of machinegun pickup drop

#### Combat Tempo
- `PISTOL_COOLDOWN`: Time between pistol shots
- `MACHINEGUN_COOLDOWN`: Time between machinegun shots

**Why grouped?**
- Reduces complexity (5 groups vs 12 individual parameters)
- Ensures coherent adjustments (all enemy pressure parameters change together)
- Makes the action space more interpretable

### 4. Reward Function

The reward function encourages maintaining a "flow zone" (balanced challenge):

```python
# Survival
+1.0 if player alive
-5.0 if player dies

# Health balance (target: 0.3 to 0.7)
+2.0 if in target zone
-penalty proportional to distance if outside

# Zombie count balance (target: 3-10)
+0.5 if in target range
-penalty if too few (too easy) or too many (too hard)

# Damage taken penalty
-1.0 if too much damage taken recently

# Kill rate bonus
+0.5 if active engagement (1-5 kills per 5 seconds)
```

**Why this reward?**
- Encourages balanced challenge, not just survival
- Penalizes both extreme ease and extreme difficulty
- Rewards active engagement (not passive survival)

### 5. Temporal Control

- **Action Interval**: DDA agent acts every 300 frames (5 seconds at 60 FPS)
- **Smoothing**: Only 30% of adjustment is applied per step to prevent oscillations
- **Episode Length**: Fixed-length episodes (60 seconds) with rolling windows

**Why fixed intervals?**
- Stability: prevents rapid oscillations
- Efficiency: reduces computational overhead
- Interpretability: easier to understand agent behavior

## Implementation Details

### State Extractor

The `DDAStateExtractor` class:
- Maintains sliding window buffers for each metric
- Computes aggregated statistics over 10-second windows
- Normalizes all features to [0, 1]

### Difficulty Manager

The `DifficultyManager` class:
- Stores base parameters and current parameters
- Applies adjustments with bounds checking
- Groups parameters for coherent adjustments
- Uses smoothing to prevent oscillations

### DDA Environment

The `DDAEnvironment` class:
- Wraps the game environment
- Tracks metrics from game steps
- Extracts DDA state
- Applies difficulty adjustments
- Computes DDA rewards

### DDA Agent

The `DDAAgent` class:
- Uses PPO algorithm (same as player agent)
- Simple actor-critic network
- Trains against the existing player agent

## Training Process

1. **Load Player Agent**: Load the trained player agent that will play the game
2. **Create DDA Environment**: Wrap the game environment with DDA functionality
3. **Train DDA Agent**: Use PPO to train the DDA agent
4. **Evaluate**: The DDA agent learns to adjust difficulty to maintain balance

## Usage Example

```python
# Load trained player agent
player_agent = PPOAgent(ppo_config)
player_agent.load("checkpoints/best_model.pth")

# Create DDA environment
dda_env = DDAEnvironment(game_env, base_params, action_interval=300)
dda_env.set_player_agent(player_agent)

# Load or train DDA agent
dda_agent = DDAAgent(dda_config)
dda_agent.load("checkpoints/dda/best_dda_model.pth")

# Run with DDA
state = dda_env.reset()
while True:
    action = dda_agent.select_action(state, deterministic=True)[0]
    state, reward, done, info = dda_env.step(action)
    if done:
        break
```

## Design Choices Summary

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| State Space | Short-term aggregated metrics | Maintains Markov property, captures current situation |
| Action Space | 5 discrete actions | Simple, interpretable, high-level control |
| Parameter Grouping | 5 groups | Reduces complexity, ensures coherence |
| Temporal Control | Fixed 5-second intervals | Stability, efficiency, interpretability |
| Smoothing | 30% per step | Prevents oscillations |
| Reward Function | Balanced challenge focus | Encourages flow zone, not just survival |
| Algorithm | PPO | Same as player agent, proven stability |

## Future Improvements

1. **Adaptive Intervals**: Adjust action interval based on game state
2. **Player Skill Estimation**: Explicitly model player skill level
3. **Multi-Objective**: Balance multiple objectives (engagement, challenge, fun)
4. **Online Learning**: Adapt to player behavior in real-time
5. **Interpretability**: Add visualization of difficulty adjustments

## Conclusion

The DDA system provides a clean, modular approach to dynamic difficulty adjustment. It maintains separation of concerns, uses appropriate abstractions, and focuses on maintaining balanced challenge rather than optimizing for a single metric.

