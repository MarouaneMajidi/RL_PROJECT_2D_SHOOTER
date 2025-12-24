# Critical Training Problems and Fixes

## Problem 1: Policy Collapse (Training Stopped)

**Symptoms from metrics:**
- `entropy: 8.8e-08` (essentially 0 - policy is deterministic)
- `clip_fraction: 0.0` (nothing being clipped - policy not updating)
- `kl_divergence: 2.6e-15` (essentially 0 - no policy change)
- All values identical across episodes (training frozen)

**Cause:**
- Entropy coefficient (`0.01`) is too low
- Policy became too deterministic too early
- Can no longer explore or learn

**Fix:**
1. Increase entropy coefficient from `0.01` to `0.05` or `0.1`
2. This keeps policy more exploratory, prevents premature collapse

## Problem 2: Still Camping Despite Idle Penalty

**Why it's still camping:**
1. `-0.3` idle penalty is still too weak when zombies are far
2. No progressive penalty (same penalty whether idle 1 second or 10 seconds)
3. No penalty for edge camping (staying near arena boundaries)
4. No zombie-density-based incentives (doesn't matter if 1 or 10 zombies coming)

**What we need:**
1. **Progressive idle penalty** - increases over time
2. **Edge camping penalty** - penalize being near arena edges
3. **Zombie-density movement reward** - stronger movement incentive when many zombies
4. **Movement bonus** - reward active movement, especially when many enemies nearby
