# Complete Reward Summary for CNN Training

## Rewards Used (from PPOConfig)

### Movement Incentives ✅

**1. Idle Penalty (Progressive)**
- Base: `-0.3` per frame when not moving
- Progressive: Increases to `-0.75` after 5 seconds, `-1.8` after 10 seconds
- **Net when idle:** `-0.2` to `-1.7` per frame (negative!)

**2. Edge Camping Penalty**
- `-0.5` per frame when stationary near arena edges
- Specifically targets top-edge camping

**3. Standing Still with Zombies Nearby**
- Base: `-1.5` when zombies within 200px and stationary
- Scales up: `-3.0` when 5+ zombies nearby (2x multiplier)
- **Net:** Up to `-3.0` per frame when camping with many zombies!

**4. Movement Bonus with Many Zombies**
- `+0.1` to `+0.5` per frame when moving AND 3+ zombies nearby
- Encourages kiting/maneuvering

### Positive Rewards for Moving ✅

**1. Survival Reward**
- `+0.1` per frame (always, regardless of movement)

**2. Shooting While Moving**
- `+0.3` per frame when shooting AND moving simultaneously

**3. Shooting at Enemies**
- `+0.2` per frame when shooting with enemies in range

**4. Moving Away from Zombies**
- `+0.5` when moving away from nearby zombies

### Negative Rewards (Always Apply)

**1. Damage Taken**
- `-5.0` per point of damage

**2. Death**
- `-20.0` when health reaches 0

## Reward Comparison

### When Camping (Many Zombies):
```
+0.1   (survival)
-0.75  (progressive idle, after 5 sec)
-0.5   (edge camping, if at top)
-3.0   (standing still with 5+ zombies)
───────────────────────────────
= -4.15 per frame (VERY BAD!)
```

### When Moving (Many Zombies):
```
+0.1   (survival)
+0.5   (movement bonus with 5+ zombies)
+0.3   (shooting while moving)
+0.2   (shooting at enemies)
+0.5   (moving away from zombies)
───────────────────────────────
= +1.6 per frame (GOOD!)
```

## Analysis

The rewards **ARE correctly configured** and **SHOULD encourage movement**.

The problem is likely:
1. **Policy already learned to camp** (from previous training)
2. **Value function is broken** (value loss = 186, explained variance = negative)
3. **Agent can't learn** because policy collapsed (entropy ≈ 0)

## Solution

**Start completely fresh training** with:
- ✅ New entropy_coef = 0.15 (prevents collapse)
- ✅ All reward fixes in place
- ✅ No checkpoint loading

The rewards are correct - the agent just needs to learn from scratch with the new reward structure!
