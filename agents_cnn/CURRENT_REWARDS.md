# Current Reward Structure for CNN Player Agent

## Summary

The current reward system has various components, but the **idle penalty is too weak** to prevent camping behavior.

## Current Reward Components

### 1. SURVIVAL REWARDS
- **Survival per step**: `+0.1` per frame (every frame the agent stays alive)
- **Death penalty**: `-20.0` when health reaches 0

### 2. IDLE/MOVEMENT PENALTIES (Current - TOO WEAK)
- **Idle penalty**: `-0.1` per frame when position unchanged
  - ⚠️ **Problem**: This is canceled out by survival reward (+0.1), so net effect is 0!
- **Standing still penalty**: `-1.5` when standing still AND zombies within 200 pixels
  - ⚠️ **Problem**: Only applies when zombies are close - agent can camp far from zombies

### 3. COMBAT REWARDS
- **Zombie kill**: `+10.0` per zombie killed
- **Strong zombie bonus**: `+4.0` bonus for killing strong zombie
- **Hit reward**: `+0.1` per successful bullet hit
- **Machine gun fire rate**: `+0.3` for sustained fire (12+ shots per second)
- **Shooting while moving**: `+0.3` reward (encourages movement + shooting)
- **Shooting at enemies**: `+0.2` when shooting with enemies in range
- **Running away without shooting**: `-0.5` penalty (when moving but not shooting with enemies nearby)
- **Shooting at out-of-bounds**: `-0.3` penalty (wasting ammo)

### 4. DAMAGE PENALTIES
- **Damage taken**: `-5.0` per point of damage (scaled by damage amount)
- **Too close to zombies**: `-0.5` when within 50 pixels of zombie

### 5. POSITIONING REWARDS
- **Moving away from zombies**: `+0.5` when moving away from nearby zombies
- **Escape danger zone**: `+1.0` when escaping high-danger situation

### 6. DANGER-BASED REWARDS
- **Danger decrease**: `+0.3` when danger score decreases
- **Danger increase**: `-0.3` when danger score increases
- **High density penalty**: `-1.0` when too many zombies in danger radius (200px)

### 7. PICKUP REWARDS
- **Health pickup**: `+3.0` base (scaled by urgency, can be up to `+9.0`)
- **Machinegun pickup**: `+5.0` base (scaled up to `+12.5`)
- **Ammo pickup**: `+2.0` base (scaled by urgency)
- **Strategic pickup bonuses**: Additional `+1.5` to `+3.0` for context-aware pickups
- **Moving toward pickups**: `+0.15` to `+0.3` per frame when moving toward pickups
- **Pickup proximity**: `+0.1` when near pickups

### 8. WEAPON REWARDS
- **Weapon switch**: `+1.0` for switching to machinegun, `+0.5` for switching to pistol

## The Problem: Why Camping is Rewarded

**Current Net Effect for Camping:**
```
Per frame when camping (no movement, shooting):
+0.1  (survival reward)
-0.1  (idle penalty)
+0.2  (shooting at enemies, if zombies in range)
+0.1  (hit reward, if hitting)
= +0.3 per frame (POSITIVE!)
```

**When zombies are far (>200px away):**
```
+0.1  (survival)
-0.1  (idle penalty)
+0.2  (shooting at enemies)
= +0.2 per frame (POSITIVE, no standing still penalty!)
```

**The Issues:**
1. ❌ Idle penalty (`-0.1`) is exactly canceled by survival reward (`+0.1`)
2. ❌ Standing still penalty only triggers when zombies are close (<200px)
3. ❌ No progressive penalty for staying in the same area for extended periods
4. ❌ No penalty for being near arena edges (where camping often occurs)

## Current Movement Detection

The current system tracks movement using:
- `self.last_player_pos = (x, y)` - compares pixel coordinates
- `current_pos == self.last_player_pos` - checks if exact same position

**This requires access to player coordinates**, which CNN agents don't have!

## What We Need for CNN Agents

Since CNN agents only have **image frames**, we need to:
1. ✅ Detect player sprite movement by comparing consecutive frames
2. ✅ Track how long the player has been stationary
3. ✅ Apply progressive penalties (increasing over time)
4. ✅ Detect edge camping (player near arena boundaries using image analysis)
