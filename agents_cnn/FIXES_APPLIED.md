# Fixes Applied to Address Training Issues

## Problem 1: Policy Collapse Fixed ✅

**Issue:** Policy became deterministic (entropy ≈ 0), training stopped

**Fix:**
- Increased `entropy_coef` from `0.01` to `0.05` in `CNNPlayerConfig`
- This prevents premature policy collapse
- Keeps policy more exploratory for longer

## Problem 2: Movement Incentives Enhanced ✅

### 2a. Progressive Idle Penalty ✅
**Before:** Constant `-0.3` penalty per frame (same whether idle 1 second or 10 seconds)

**After:** Progressive penalty that increases over time
```python
idle_seconds = idle_frames / 60.0
progressive_penalty = -0.3 * (1.0 + idle_seconds * 0.5)
```

**Effect:**
- First second: `-0.3` per frame
- After 2 seconds: `-0.45` per frame
- After 5 seconds: `-0.75` per frame
- After 10 seconds: `-1.8` per frame

This makes extended camping increasingly expensive!

### 2b. Edge Camping Penalty ✅
**New:** Penalty for staying near arena edges (where camping often occurs)

- Detects if player is within 100 pixels of any edge
- Applies `-0.5` penalty when stationary near edges
- Specifically targets the "stay at top" behavior

### 2c. Zombie-Density-Based Rewards ✅
**New:** Movement rewards that scale with zombie count

1. **Standing still penalty scales up:**
   - Base: `-1.5` when 1 zombie nearby
   - Scales up to `-3.0` when 5+ zombies nearby
   - Formula: `penalty * min(zombie_count / 5.0, 2.0)`

2. **Movement bonus when many zombies:**
   - When 3+ zombies nearby AND moving: `+0.1` to `+0.5` bonus per frame
   - Formula: `min(zombie_count * 0.1, 0.5)`
   - Encourages kiting/maneuvering when overwhelmed

## New Reward Structure

### When Camping (Many Zombies):
```
+0.1   (survival)
-0.3   (base idle penalty, progressive if long)
-0.5   (edge camping penalty, if near edge)
-3.0   (standing still with 5+ zombies: -1.5 * 2.0)
───────────────────────────────
= -3.7 per frame (STRONGLY DISCOURAGED!)
```

### When Moving (Many Zombies):
```
+0.1   (survival)
+0.5   (movement bonus with 5+ zombies)
+0.3   (shooting while moving)
+0.2   (shooting at enemies)
───────────────────────────────
= +1.1 per frame (STRONGLY ENCOURAGED!)
```

## What to Do Next

1. **Start fresh training** (don't continue from collapsed model):
   ```bash
   python scripts_cnn/train_player_cnn.py --timesteps 1000000
   ```

2. **Monitor training:**
   - Watch `entropy` - should stay above 0.001 (not collapse to 0)
   - Watch `episode_score` - should increase (more kills)
   - Watch `episode_length` - should increase (surviving longer)

3. **Expected improvements:**
   - Agent should move more dynamically
   - Less camping at edges
   - Better kiting when many zombies
   - Training should continue learning (not collapse)

## Configuration Changes Summary

**agents_cnn/player_cnn/config.py:**
- `entropy_coef: 0.01 → 0.05` (5x increase)

**agents/ppo_agent/config.py:**
- Added `reward_edge_camping_penalty: float = -0.5`

**agents/ppo_agent/env.py:**
- Progressive idle penalty (increases over time)
- Edge camping detection and penalty
- Zombie-density-scaled penalties
- Movement bonus when many zombies nearby
