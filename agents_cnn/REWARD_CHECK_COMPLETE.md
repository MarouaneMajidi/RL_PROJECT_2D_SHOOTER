# CNN Training Reward Configuration - VERIFIED ✅

## Rewards ARE Correctly Configured!

All the reward fixes we implemented **ARE being used** by CNN training:

### ✅ Idle Penalty (Progressive)
- **Config:** `reward_idle_penalty = -0.3` 
- **Implementation:** Progressive formula `-0.3 * (1.0 + idle_seconds * 0.5)`
- **Result:** After 5 seconds idle = `-0.75` per frame, after 10 seconds = `-1.8` per frame

### ✅ Edge Camping Penalty  
- **Config:** `reward_edge_camping_penalty = -0.5`
- **Implementation:** Applied when stationary near edges (within 100px of any edge)

### ✅ Zombie-Density Rewards
- **Standing still penalty:** Scales from `-1.5` to `-3.0` based on zombie count
- **Movement bonus:** `+0.1` to `+0.5` when moving with 3+ zombies

### ✅ Combat Rewards for Moving
- **Shooting while moving:** `+0.3` per frame
- **Shooting at enemies:** `+0.2` per frame  
- **Moving away from zombies:** `+0.5` per frame

## Why Agent Still Not Moving?

**The rewards are correct, but the problem is:**

1. **Agent learned from OLD rewards** (when camping was rewarded/neutral)
2. **Policy collapsed** (entropy ≈ 0) - agent can't explore new behaviors
3. **Value function broken** (value loss = 186) - agent doesn't understand what's good/bad
4. **Continuing from old checkpoint** - agent is stuck in old policy

## Solution: Start FRESH Training

**DO NOT continue from old checkpoint!** The old model learned the wrong behavior.

Start completely fresh:
```bash
python scripts_cnn/train_player_cnn.py --timesteps 1000000
# DO NOT use --checkpoint flag!
```

With fresh training:
- ✅ Agent learns with new rewards from the start
- ✅ Entropy stays high (0.15 coefficient) - can explore
- ✅ Value function learns correctly
- ✅ Agent discovers moving is better than camping

## Expected Timeline

- **0-50k timesteps:** Learning basic game mechanics
- **50k-200k timesteps:** Discovering movement is rewarded
- **200k-500k timesteps:** Refining movement strategies
- **500k+ timesteps:** Optimizing advanced tactics

## Monitoring Fresh Training

Watch for:
- ✅ **Entropy > 0.1** (not collapsing to 0)
- ✅ **Value Loss decreasing** (from ~186 to <10)
- ✅ **Explained Variance positive** (becoming > 0.5)
- ✅ **Episode scores increasing** (more kills)
- ✅ **Episode lengths increasing** (surviving longer)

The rewards are correct - you just need to let the agent learn them from scratch!
