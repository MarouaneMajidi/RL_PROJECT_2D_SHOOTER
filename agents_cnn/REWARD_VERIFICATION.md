# CNN Training Reward Configuration Verification

## What CNN Training Uses

The CNN training script (`scripts_cnn/train_player_cnn.py`) uses:
1. **CNNPlayerConfig** - for CNN agent hyperparameters (learning rate, entropy_coef, etc.)
2. **PPOConfig** - for environment rewards (this is what matters for movement incentives!)

The environment rewards come from `PPOConfig` because:
- Line 97: `ppo_config = PPOConfig()`
- Line 101: `env = CNNEnvWrapper(ppo_config, ...)`
- The wrapper creates `ZombieShooterEnv(config)` which uses rewards from `PPOConfig`

## Current Reward Configuration (PPOConfig)

### Idle Penalty ✅
- **Config:** `reward_idle_penalty: float = -0.3` ✅ (updated from -0.1)
- **Implementation:** Progressive penalty in `env.py` ✅
- **Formula:** `progressive_penalty = -0.3 * (1.0 + idle_seconds * 0.5)`

### Edge Camping Penalty ✅
- **Config:** `reward_edge_camping_penalty: float = -0.5` ✅ (exists in config)
- **Implementation:** Detects edges and applies penalty when stationary ✅

### Zombie-Density Rewards ✅
- **Standing still penalty scales:** Up to 2x when many zombies ✅
- **Movement bonus:** +0.1 to +0.5 when 3+ zombies and moving ✅

## Why Agent Still Not Moving?

Possible reasons:

1. **Rewards are still negative overall**
   - Even with penalties, if agent isn't moving it gets negative rewards
   - But moving should give positive rewards (shooting while moving, etc.)

2. **Value function might be broken**
   - High value loss (186) suggests agent doesn't understand rewards
   - Might need more training to learn value function

3. **Policy needs more exploration**
   - Even with entropy_coef = 0.15, might need more
   - Agent might be stuck in local minimum

4. **Reward balance issue**
   - The penalties might be too strong relative to positive rewards
   - Agent might see all actions as bad

Let me check if all the rewards are actually being applied correctly.
