# Data Collected for DDA Evaluation

This document explains all metrics collected during DDA agent evaluation.

## Overview

The evaluation script runs 100 games per DDA agent and collects metrics at two levels:
1. **Per 5-second interval** - Detailed metrics for each DDA action period
2. **Per game** - Aggregated metrics summarizing the entire game session

---

## PER 5-SECOND INTERVAL METRICS

These metrics are collected every 5 seconds (when the DDA agent makes a decision).

### 1. Action Taken (action_name)
**Type:** String  
**Values:** 
- "Do Nothing"
- "Slightly Easier"
- "Much Easier"
- "Slightly Harder"
- "Much Harder"

**Explanation:**  
The DDA action chosen by the agent at this interval. This shows the agent's decision-making behavior and how it responds to game state.

**Why it matters:**  
- Shows if the agent is actively adjusting difficulty or staying passive
- Reveals patterns in decision-making (e.g., always choosing "easier" vs balanced choices)
- Helps identify if the agent is oscillating between easy/hard

---

### 2. Health Percentage at Start (health_start)
**Type:** Float (0.0 - 100.0)  
**Unit:** Percentage

**Explanation:**  
Player's health percentage at the beginning of the 5-second interval, right before the DDA action is taken.

**Why it matters:**  
- Shows the game state that triggered the DDA decision
- Allows correlation between health level and DDA action choice
- Helps understand if the agent responds appropriately to low/high health

---

### 3. Health Percentage at End (health_end)
**Type:** Float (0.0 - 100.0)  
**Unit:** Percentage

**Explanation:**  
Player's health percentage at the end of the 5-second interval, after the DDA action has been applied.

**Why it matters:**  
- Shows the impact of the DDA action on player health
- Allows calculation of health change during the interval (health_end - health_start)
- Helps evaluate if DDA actions are effective

---

### 4. Kills in Interval (kills_interval)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of zombies killed during this 5-second interval.

**Why it matters:**  
- Measures combat effectiveness during the interval
- Shows if difficulty adjustments affect kill rate
- Indicates player engagement and combat activity

---

### 5. Damage Taken in Interval (damage_taken_interval)
**Type:** Float  
**Unit:** Health points

**Explanation:**  
Total damage taken by the player during this 5-second interval.

**Why it matters:**  
- Measures pressure and threat level
- Shows if DDA adjustments reduce or increase incoming damage
- Correlates with difficulty level (higher damage = harder)

---

### 6. Damage Dealt in Interval (damage_dealt_interval)
**Type:** Float  
**Unit:** Health points

**Explanation:**  
Total damage dealt by the player to zombies during this 5-second interval.

**Why it matters:**  
- Measures offensive effectiveness
- Shows combat engagement level
- Helps evaluate if difficulty affects player's ability to deal damage

---

### 7. Zombies Alive at Start (zombies_alive_start)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of zombies alive at the beginning of the 5-second interval.

**Why it matters:**  
- Shows enemy pressure level when DDA decision is made
- Indicates game difficulty state (more zombies = harder)
- Helps understand what the DDA agent is responding to

---

## PER GAME METRICS

These metrics summarize the entire game session from start to death.

### 1. Total Survival Time (survival_time_steps)
**Type:** Integer  
**Unit:** Game steps (frames)

**Explanation:**  
Total number of game steps (frames) the player survived before death. At 60 FPS, 1 step = 1/60 second.

**Why it matters:**  
- Primary metric for evaluating DDA effectiveness
- Longer survival = better DDA performance (if maintaining challenge)
- Allows comparison between different DDA agents
- Can be converted to seconds: `survival_time_seconds = survival_time_steps / 60.0`

---

### 2. Total Kills (total_kills)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Total number of zombies killed during the entire game session.

**Why it matters:**  
- Measures overall combat performance
- Higher kills with similar survival time = more engaging gameplay
- Shows if DDA maintains active combat (not just survival)

---

### 3. Average Health Percentage (avg_health_percent)
**Type:** Float  
**Unit:** Percentage (0.0 - 100.0)

**Explanation:**  
Mean health percentage across all frames of the game session.

**Why it matters:**  
- Indicates overall difficulty balance
- Target range: 30-70% (flow zone)
- Too low (<30%) = too hard, too high (>70%) = too easy
- Shows if DDA maintains balanced challenge

---

### 4. Health Standard Deviation (health_std)
**Type:** Float  
**Unit:** Percentage

**Explanation:**  
Standard deviation of health percentage across all frames. Measures health stability.

**Why it matters:**  
- Low std = stable health (good DDA)
- High std = oscillating health (poor DDA, creates instability)
- Target: Lower is better (smooth experience)
- Indicates if DDA prevents health swings

---

### 5. Minimum Health (min_health)
**Type:** Float  
**Unit:** Percentage (0.0 - 100.0)

**Explanation:**  
The lowest health percentage reached during the game session.

**Why it matters:**  
- Shows how close to death the player came
- Lower values = more dangerous situations
- Helps evaluate if DDA prevents near-death scenarios
- Combined with near-death count, shows danger frequency

---

### 6. Time in Flow Zone (time_in_flow_zone_percent)
**Type:** Float  
**Unit:** Percentage (0.0 - 100.0)

**Explanation:**  
Percentage of game time spent with health between 30% and 70%. This is the "optimal challenge zone."

**Why it matters:**  
- **Primary DDA objective metric**
- Higher = better DDA performance
- Target: >60% of time in flow zone
- Indicates if DDA successfully maintains balanced challenge

---

### 7. Time in Danger Zone (time_in_danger_zone_percent)
**Type:** Float  
**Unit:** Percentage (0.0 - 100.0)

**Explanation:**  
Percentage of game time spent with health below 30% (dangerous/threatening situations).

**Why it matters:**  
- Lower = better DDA performance
- Shows if DDA reduces stressful situations
- Target: <20% of time in danger zone
- High values indicate DDA is not preventing difficulty spikes

---

### 8. Time in Safety Zone (time_in_safety_zone_percent)
**Type:** Float  
**Unit:** Percentage (0.0 - 100.0)

**Explanation:**  
Percentage of game time spent with health above 70% (too easy, low challenge).

**Why it matters:**  
- Lower = better DDA performance (maintains challenge)
- Shows if DDA prevents the game from being too easy
- Target: <30% of time in safety zone
- High values indicate DDA is not maintaining sufficient challenge

---

### 9. Near-Death Moments Count (near_death_count)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the player's health dropped below 20% during the game session.

**Why it matters:**  
- Measures frequency of critical situations
- Lower = better (fewer near-death experiences)
- Shows if DDA prevents dangerous health drops
- Each count represents a moment of high stress/frustration

---

### 10. Near-Death Time Percentage (near_death_time_percent)
**Type:** Float  
**Unit:** Percentage (0.0 - 100.0)

**Explanation:**  
Percentage of game time spent with health below 20% (critical danger).

**Why it matters:**  
- More comprehensive than count (considers duration, not just occurrences)
- Lower = better DDA performance
- Target: <10% of time in near-death
- Shows both frequency and duration of critical situations

---

### 11. Action Distribution

These metrics count how many times each DDA action was chosen during the game.

#### 11a. Do Nothing Count (action_do_nothing_count)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the DDA agent chose "Do Nothing" (action 0) during the game.

**Why it matters:**  
- High count = agent finds game balanced (good sign)
- Very high count = agent might be too passive
- Very low count = agent is always adjusting (might indicate instability)

---

#### 11b. Slightly Easier Count (action_slightly_easier_count)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the DDA agent chose "Slightly Easier" (action 1) during the game.

**Why it matters:**  
- Shows frequency of small difficulty reductions
- Balanced with harder actions = good DDA behavior
- Always choosing easier = agent thinks game is too hard

---

#### 11c. Much Easier Count (action_much_easier_count)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the DDA agent chose "Much Easier" (action 2) during the game.

**Why it matters:**  
- Shows frequency of large difficulty reductions
- High count = agent frequently finds game too difficult
- Should be used sparingly (emergency situations)

---

#### 11d. Slightly Harder Count (action_slightly_harder_count)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the DDA agent chose "Slightly Harder" (action 3) during the game.

**Why it matters:**  
- Shows frequency of small difficulty increases
- Balanced with easier actions = good DDA behavior
- Always choosing harder = agent thinks game is too easy

---

#### 11e. Much Harder Count (action_much_harder_count)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the DDA agent chose "Much Harder" (action 4) during the game.

**Why it matters:**  
- Shows frequency of large difficulty increases
- High count = agent frequently finds game too easy
- Should be used sparingly (when game is clearly too easy)

---

### 12. Action Oscillations (action_oscillations)
**Type:** Integer  
**Unit:** Count

**Explanation:**  
Number of times the DDA agent changed direction between easier and harder actions consecutively (e.g., Easier → Harder → Easier, or Harder → Easier → Harder).

**Why it matters:**  
- Lower = better (stable adjustments)
- High count = agent is oscillating (poor stability)
- Indicates if DDA creates smooth or jerky difficulty changes
- Target: <5 oscillations per game (for 60-second average game)

---

## METRIC INTERPRETATION GUIDE

### Good DDA Performance Indicators:
- ✅ **Survival time:** Longer or similar to baseline
- ✅ **Time in flow zone:** >60% of game time
- ✅ **Time in danger zone:** <20% of game time
- ✅ **Health std:** Low (<15%)
- ✅ **Action distribution:** Balanced (not always one action)
- ✅ **Action oscillations:** Low (<5 per game)

### Poor DDA Performance Indicators:
- ❌ **Survival time:** Much shorter than baseline
- ❌ **Time in flow zone:** <40% of game time
- ❌ **Time in danger zone:** >30% of game time
- ❌ **Health std:** High (>20%)
- ❌ **Action distribution:** Always one action (e.g., always "easier")
- ❌ **Action oscillations:** High (>10 per game)

---

## DATA STRUCTURE

### Interval-Level CSV Structure:
```
game_id,interval_number,action_name,health_start,health_end,kills_interval,damage_taken_interval,damage_dealt_interval,zombies_alive_start
```

### Game-Level CSV Structure:
```
dda_agent_name,game_id,survival_time_steps,total_kills,avg_health_percent,health_std,min_health,time_in_flow_zone_percent,time_in_danger_zone_percent,time_in_safety_zone_percent,near_death_count,near_death_time_percent,action_do_nothing_count,action_slightly_easier_count,action_much_easier_count,action_slightly_harder_count,action_much_harder_count,action_oscillations
```

---

## SUMMARY STATISTICS

For each DDA agent, the evaluation script also calculates:
- **Mean** - Average across 100 games
- **Std** - Standard deviation across 100 games
- **Min** - Minimum value across 100 games
- **Max** - Maximum value across 100 games

These statistics help compare different DDA agents and identify the best performing one.

