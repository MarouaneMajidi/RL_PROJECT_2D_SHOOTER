# Analysis: Will Increasing Idle Penalty Encourage Movement?

## Current Situation

**Current Rewards Per Frame When Idle:**
```
+0.1  (reward_survival_per_step)
-0.1  (reward_idle_penalty)
─────────────────────────────
= 0.0  (NET: No penalty for staying still!)
```

**Current Rewards Per Frame When Moving:**
```
+0.1  (reward_survival_per_step)
+0.3  (reward_shooting_while_moving, if shooting)
+0.2  (reward_shooting_at_enemies, if enemies in range)
─────────────────────────────
= +0.6 per frame (POSITIVE reward for moving + shooting)
```

## If We Increase Idle Penalty

### Option 1: Double the Idle Penalty (-0.2)
**When Idle:**
```
+0.1  (survival)
-0.2  (idle penalty)
─────────────────────────────
= -0.1 per frame (NET: Small negative penalty)
```

**When Moving:**
```
+0.1  (survival)
+0.3  (shooting while moving)
+0.2  (shooting at enemies)
─────────────────────────────
= +0.6 per frame (NET: Still positive)
```

**Result:** ✅ Movement becomes more attractive, but penalty is still small

### Option 2: Triple the Idle Penalty (-0.3)
**When Idle:**
```
+0.1  (survival)
-0.3  (idle penalty)
─────────────────────────────
= -0.2 per frame (NET: Moderate negative penalty)
```

**Result:** ✅✅ Better! Creates stronger incentive to move

### Option 3: Quadruple the Idle Penalty (-0.4)
**When Idle:**
```
+0.1  (survival)
-0.4  (idle penalty)
─────────────────────────────
= -0.3 per frame (NET: Strong negative penalty)
```

**Result:** ✅✅✅ Strong incentive, but may be too harsh

## Important Note

**Good News:** The current position-based idle detection (`current_pos == self.last_player_pos`) **DOES work for CNN agents** because:

1. The CNN wrapper calls `env.step(action)` which calculates rewards using position
2. The agent doesn't see positions in observations, but rewards are still calculated correctly
3. So increasing `reward_idle_penalty` will affect CNN training

## Recommendations

### Quick Fix (Simplest)
**Increase idle penalty to `-0.3` or `-0.4`:**
```python
reward_idle_penalty: float = -0.3  # or -0.4
```

This creates a net negative reward when idle (-0.2 to -0.3 per frame), making movement more attractive.

### Better Solution (Progressive Penalty)
Instead of constant penalty, use **progressive penalty** that increases over time:

```python
# Current (constant)
if idle:
    reward += -0.1

# Better (progressive)
if idle:
    idle_time = self.idle_frames / 60.0  # Convert to seconds
    penalty = -0.1 * (1 + idle_time)  # Increases over time
    reward += penalty
```

This means:
- First second idle: -0.1 per frame
- After 2 seconds: -0.2 per frame  
- After 5 seconds: -0.5 per frame
- After 10 seconds: -1.0 per frame

### Best Solution (Image-Based + Progressive)
Since you want image-based detection for CNN agents:

1. Detect movement by comparing consecutive frames (image-based)
2. Apply progressive penalty that increases over time
3. This works purely from images (no position data needed)

## My Recommendation

**Start with a simple increase to `-0.3` or `-0.4`** - this is the quickest fix and will definitely help. Then we can implement progressive/image-based penalties later if needed.

The current `-0.1` is definitely too weak because it's completely canceled by the survival reward.
