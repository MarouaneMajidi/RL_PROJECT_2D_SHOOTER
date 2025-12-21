# RL Agent Fixes & Improvements

## Overview

This document describes all fixes and improvements made to the RL agent and game logic.

---

## 1. Health Pack Pickup Logic - FIXED ✅

### Problem
The agent wasn't prioritizing health pickups when at low health, and the observation space didn't include information about pickups.

### Solution

#### A. Enhanced Observation Space
Added pickup detection to the state representation:

```python
# In ppo_agent/env.py - _get_state()

# Nearest health pack info (3 values): [distance, angle, exists]
nearest_health = self._get_nearest_pickup('health')
if nearest_health:
    health_dist = self._distance(self.player['x'], self.player['y'], 
                                nearest_health['x'], nearest_health['y'])
    health_angle = self._angle_to(self.player['x'], self.player['y'],
                                 nearest_health['x'], nearest_health['y'])
    state.extend([
        min(health_dist / 1000.0, 1.0),
        health_angle / 360.0,
        1.0  # Exists
    ])
else:
    state.extend([1.0, 0.0, 0.0])  # No health pack
```

#### B. Danger Level Metric
Added a danger level to help the agent understand when it needs health:

```python
def _compute_danger_level(self) -> float:
    """
    Compute danger from 0.0 (safe) to 1.0 (critical) based on:
    - Low health increases danger (40% weight)
    - Nearby zombies increase danger (40% weight)
    - Very close zombie is critical (20% weight)
    """
    danger = 0.0
    health_danger = 1.0 - (self.player['health'] / 100.0)
    danger += health_danger * 0.4
    
    nearby_zombies = sum(1 for z in self.zombies 
                        if self._distance(...) < 150)
    zombie_danger = min(nearby_zombies / 5.0, 1.0)
    danger += zombie_danger * 0.4
    
    if self.zombies:
        min_dist = min(self._distance(...) for z in self.zombies)
        if min_dist < 50:
            danger += 0.2
    
    return min(danger, 1.0)
```

#### C. Reward Shaping for Health Pickups
Added incentives to collect health when low:

```python
# In ppo_agent/config.py
reward_health_pickup: float = 8.0          # Base reward (scales with health gained)
reward_low_health_near_pickup: float = 0.1  # Reward per step when near health
reward_ignore_health_penalty: float = -0.05 # Penalty for ignoring health when hurt

# In ppo_agent/env.py - step()
if self.player['health'] < 50:  # Low health threshold
    nearest_health = self._get_nearest_pickup('health')
    if nearest_health is not None:
        health_dist = self._distance(...)
        if health_dist < 200:  # Health pack is close
            # Reward for being near health pack when low on health
            reward += config.reward_low_health_near_pickup * (1.0 - health_dist / 200)
        else:
            # Small penalty for ignoring health pack when low on health
            reward += config.reward_ignore_health_penalty
```

---

## 2. Machine Gun Rapid-Fire - FIXED ✅

### Problem
The agent was picking up the machine gun but using it like a pistol (single shots instead of continuous fire).

### Root Cause
1. Action space only had 6 actions (no weapon switching)
2. Environment hardcoded pistol usage
3. No reward incentive for machine gun's higher DPS

### Solution

#### A. Extended Action Space (6 → 8 actions)

```python
# In ppo_agent/config.py
action_dim: int = 8  # Extended from 6 to 8

# Action mapping:
# 0: Move Up
# 1: Move Down
# 2: Move Left
# 3: Move Right
# 4: Shoot (continuous fire)
# 5: Idle (no action)
# 6: Switch to Pistol      # NEW
# 7: Switch to Machine Gun  # NEW
```

#### B. Weapon State in Observations

```python
# In ppo_agent/env.py - _get_state()
# Player state now includes (12 values):
state.extend([
    self.player['x'] / self.SCREEN_WIDTH,
    self.player['y'] / self.SCREEN_HEIGHT,
    0.0,  # vel_x
    0.0,  # vel_y
    self.player['health'] / 100.0,
    self.player['shoot_cooldown'] / self.PISTOL_COOLDOWN,
    self.player['angle'] / 360.0,
    min(len(self.zombies), self.max_zombies) / self.max_zombies,
    1.0 if self.player['has_machinegun'] else 0.0,       # NEW
    min(self.player['machinegun_ammo'] / 100.0, 1.0),    # NEW
    1.0 if self.player['weapon'] == 'machinegun' else 0.0,  # NEW
    self._compute_danger_level()                          # NEW
])
```

#### C. Proper Weapon Shooting Logic

```python
# In ppo_agent/env.py - _player_shoot()
def _player_shoot(self) -> Optional[str]:
    if self.player['shoot_cooldown'] > 0:
        return None
    
    if (self.player['weapon'] == 'machinegun' and 
        self.player['has_machinegun'] and 
        self.player['machinegun_ammo'] > 0):
        # Machine gun: faster fire rate (5 frames vs 20), higher DPS
        self.bullets.append({
            'x': self.player['x'],
            'y': self.player['y'],
            'angle': angle_rad,
            'damage': self.MACHINEGUN_DAMAGE,  # 10 damage
            'weapon': 'machinegun'
        })
        self.player['shoot_cooldown'] = self.MACHINEGUN_COOLDOWN  # 5 frames
        self.player['machinegun_ammo'] -= 1
        return 'machinegun'
    else:
        # Pistol: slower fire rate (20 frames), less DPS
        self.bullets.append({
            'x': self.player['x'],
            'y': self.player['y'],
            'angle': angle_rad,
            'damage': self.PISTOL_DAMAGE,  # 20 damage
            'weapon': 'pistol'
        })
        self.player['shoot_cooldown'] = self.PISTOL_COOLDOWN  # 20 frames
        return 'pistol'
```

#### D. DPS Comparison & Rewards

| Weapon | Damage | Cooldown | DPS | Kill Reward |
|--------|--------|----------|-----|-------------|
| Pistol | 20 | 20 frames | 1.0/frame | 10.0 |
| Machine Gun | 10 | 5 frames | 2.0/frame | 12.0 (+2 bonus) |

```python
# In ppo_agent/config.py
reward_machinegun_kill_bonus: float = 2.0   # Bonus per MG kill
reward_machinegun_dps_bonus: float = 0.5    # Bonus for sustained fire

# In ppo_agent/env.py - _update_bullets()
if zombie['health'] <= 0:
    kill_reward += self.config.reward_zombie_kill  # Base: 10.0
    
    # Bonus for machine gun kills (encourages using it)
    if bullet.get('weapon') == 'machinegun':
        kill_reward += self.config.reward_machinegun_kill_bonus  # +2.0
```

#### E. Sustained Fire Tracking

```python
# In ppo_agent/env.py - step()
# Track shots fired in last 60 frames (1 second)
if shot_result:
    self.dps_window.append((self.episode_steps, shot_result))

# Clean old entries
self.dps_window = [(t, d) for t, d in self.dps_window if self.episode_steps - t < 60]

# Reward for sustained DPS with machine gun
if len(self.dps_window) >= 3:
    mg_hits = sum(1 for _, weapon in self.dps_window if weapon == 'machinegun')
    if mg_hits >= 5:  # Good machine gun usage
        reward += self.config.reward_machinegun_dps_bonus  # +0.5
```

---

## 3. Wave Reset Bug - FIXED ✅

### Problem
When the player died on wave N and pressed R to retry:
- UI reset to wave 1
- Internal game logic continued from wave N (wrong spawn rate, wrong difficulty)

### Root Cause
1. Wave and spawn_rate were local variables, not properly reset
2. No centralized game state management
3. Wave calculation based on cumulative kills, not per-episode kills

### Solution

#### A. Created GameState Class

```python
# In main.py
class GameState:
    """
    Manages all game state in one place.
    Provides a clean resetGame() method that ensures everything is properly reset.
    """
    
    def __init__(self):
        self.player = None
        self.zombies = []
        self.bullets = []
        self.knife_attacks = []
        self.pickups = []
        self.zombies_killed = 0
        self.wave = 1
        self.spawn_timer = 0
        self.spawn_rate = SPAWN_RATE  # Initial spawn rate (60)
        self.game_over = False
        
    def resetGame(self):
        """
        COMPLETE GAME RESET - Fixes the wave reset bug.
        
        This method ensures ALL game state is reset to initial values:
        1. Player state (position, health, weapons)
        2. Enemy lists (zombies, bullets, etc.)
        3. Wave counter and spawn rate
        4. All timers and tracking variables
        """
        # Reset player
        if self.player is None:
            self.player = Player()
        else:
            self.player.reset()
        
        # Clear all game objects
        self.zombies = []
        self.bullets = []
        self.knife_attacks = []
        self.pickups = []
        
        # CRITICAL: Reset wave and spawn rate to initial values
        self.wave = 1
        self.spawn_rate = SPAWN_RATE  # Reset to 60!
        
        # Reset tracking
        self.zombies_killed = 0
        self.spawn_timer = 0
        self.game_over = False
        
        return self
```

#### B. Per-Episode Kill Tracking in Environment

```python
# In ppo_agent/env.py
def __init__(self, ...):
    self.zombies_killed = 0  # Total across all episodes
    self.zombies_killed_this_episode = 0  # Per-episode counter

def reset(self) -> np.ndarray:
    # CRITICAL: Reset ALL state including wave and spawn rate
    self.wave = 1
    self.spawn_rate = self.SPAWN_RATE  # Reset to 60
    self.zombies_killed = 0
    self.zombies_killed_this_episode = 0  # Reset per-episode counter
    
def step(self, action):
    # Wave progression based on per-episode kills
    if self.zombies_killed_this_episode > 0 and self.zombies_killed_this_episode % 10 == 0:
        expected_wave = (self.zombies_killed_this_episode // 10) + 1
        if expected_wave > self.wave:
            self.wave = expected_wave
            self.spawn_rate = max(10, self.spawn_rate - 5)
```

#### C. Usage in Game Modes

```python
# In main_manual_mode()
game = GameState()
game.resetGame()  # Initial setup

# When player dies and presses R:
if game.game_over and event.key == K_r:
    game.resetGame()  # Complete reset including wave!

# In main_agent_mode()
game = GameState()
game.resetGame()

# On episode restart:
if game.game_over and event.key == K_r:
    game.resetGame()  # Complete reset
    episode_reward = 0
    episode_steps = 0
    state = get_agent_state()
```

---

## 4. Updated PPO Configuration

### Hyperparameter Changes

```python
# ppo_agent/config.py

# Extended dimensions
action_dim: int = 8   # Was 6
state_dim: int = 108  # Was 77

# Optimized for weapon learning
learning_rate: float = 2.5e-4  # Down from 3e-4 for stability
entropy_coef: float = 0.02    # Up from 0.01 for exploration

# New reward parameters
reward_health_pickup: float = 8.0
reward_low_health_near_pickup: float = 0.1
reward_ignore_health_penalty: float = -0.05
reward_machinegun_pickup: float = 5.0
reward_machinegun_kill_bonus: float = 2.0
reward_machinegun_dps_bonus: float = 0.5
```

### State Space Breakdown (108 dimensions)

| Component | Values | Description |
|-----------|--------|-------------|
| Player | 12 | Position, health, cooldown, weapon state, danger |
| Zombies | 90 | 15 zombies × 6 features each |
| Health Pickup | 3 | Distance, angle, exists |
| MG Pickup | 3 | Distance, angle, exists |
| **Total** | **108** | |

---

## 5. Retraining Instructions

### Delete Old Model (Incompatible)
```bash
rm -rf checkpoints/
```

### Retrain with New Configuration
```bash
conda activate rl-game
python train_ppo.py --timesteps 2000000 --batch-size 32
```

### Expected Learning Progression

**After 500k timesteps:**
- Agent learns basic shooting and movement
- Starts noticing health pickups
- May occasionally switch weapons

**After 1M timesteps:**
- Consistent health pickup collection when low
- Learns machine gun is more effective
- Wave 2-3 reached regularly

**After 2M timesteps:**
- Strategic weapon switching
- Prioritizes health when injured
- Wave 3-5 reached
- Sustained machine gun fire

---

## 6. Debugging Tips

### Visualize Action Probabilities
```python
# During testing, add this to see what the agent is thinking:
from ppo_agent import PPOAgent, PPOConfig
import torch

agent = PPOAgent(PPOConfig())
agent.load("checkpoints/best_model.pth")

state = env.reset()
with torch.no_grad():
    state_tensor = torch.from_numpy(state).float().to(agent.device)
    logits, value = agent.policy(state_tensor.unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)
    
    print("Action Probabilities:")
    for i, prob in enumerate(probs[0]):
        action_names = ['Up', 'Down', 'Left', 'Right', 'Shoot', 'Idle', 'Pistol', 'MG']
        print(f"  {action_names[i]}: {prob.item():.3f}")
```

### Verify Continuous Fire
```python
# Check if MG is being used properly:
mg_shots = 0
total_shots = 0

for step in range(1000):
    action = 4  # Shoot
    state, reward, done, info = env.step(action)
    if info.get('current_weapon') == 'machinegun':
        mg_shots += 1
    total_shots += 1
    
print(f"MG shots: {mg_shots}/{total_shots} = {mg_shots/total_shots*100:.1f}%")
```

### Validate Wave Resets
```python
# Test wave reset:
env = ZombieShooterEnv(config)

# Simulate playing until wave 3
for _ in range(300):
    state, reward, done, info = env.step(4)
    if done:
        break

print(f"Before reset: Wave={env.wave}, Spawn Rate={env.spawn_rate}")

env.reset()

print(f"After reset: Wave={env.wave}, Spawn Rate={env.spawn_rate}")
# Should print: Wave=1, Spawn Rate=60
```

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `ppo_agent/config.py` | Extended state/action dims, new rewards, hyperparameter tuning |
| `ppo_agent/env.py` | Weapon support, pickup detection, wave fix, DPS tracking |
| `main.py` | GameState class, weapon switching, complete reset |
| `train_ppo.py` | Weapon logging in CSV output |

---

## 8. Summary

| Issue | Status | Solution |
|-------|--------|----------|
| Health pickup not prioritized | ✅ Fixed | Added pickup observations + reward shaping |
| Machine gun used like pistol | ✅ Fixed | Added weapon switching actions + DPS rewards |
| Wave reset bug | ✅ Fixed | Created GameState class with resetGame() |
| Agent not exploring weapons | ✅ Fixed | Increased entropy, added MG bonuses |

The agent should now learn to:
1. Collect health packs when injured
2. Switch to machine gun when available
3. Use sustained fire with MG for higher DPS
4. Progress through waves with proper resets
