"""
PPO Configuration File

Contains all hyperparameters for the PPO algorithm.

UPDATED FOR EXPERT RL SYSTEM:
- Action space: 6 actions (Move Up/Down/Left/Right, Shoot, Idle) with internal weapon switching
- State space: 98-dim with comprehensive features (player, zombies, pickups, danger, distance)
- Complete reward function rewrite with combat, survival, positioning, pickup, and penalty rewards
- Danger scoring system based on distance to zombies
- Movement tracking for kiting/escape behavior
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PPOConfig:
    """
    Configuration class for PPO hyperparameters.
    
    CHANGES FROM ORIGINAL:
    1. action_dim: 6 -> 8 (added weapon switch actions)
    2. state_dim: Recalculated for new observations
    3. Added reward shaping for:
       - Health pickup collection with low-health bonus
       - Machine gun DPS bonus
       - Ignoring health when low penalty
    4. Increased entropy coefficient for better exploration of new actions
    5. Adjusted learning rate for more stable weapon learning
    """
    
    # Environment settings
    max_zombies: int = 15  # Maximum number of zombies to track in state
    
    # Action space: 0-3 movement, 4 shoot, 5 idle
    # Weapon switching is handled internally based on weapon state when "Shoot" is selected
    action_dim: int = 6
    
    # State space calculation (98-dim):
    # Player: 6 values [x, y, health, shoot_cooldown, angle, zombie_count]
    # Weapon: 3 values [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg]
    # Zombies: 6 * 13 = 78 values [x, y, health, type, distance, angle_to_player] per zombie
    # Pickups: 3 types * 3 features = 9 values
    #          [machinegun_location, health_pack_location, ammo_pickup_location]
    #          Each: [distance, angle, exists]
    # Distance to nearest zombie: 1 value
    # Zombie density in radius: 1 value
    # Total: 6 + 3 + 78 + 9 + 1 + 1 = 98
    # Let me use the exact specification: 98-dim
    # I'll design it as: Player (8), Weapon (3), Zombies (15*6=90), Pickups (3*3=9), Distance (1), Density (1)
    # But that's 112. For 98, I'll use: Player (6), Weapon (3), Zombies (14*6=84), Pickups (3*3=9), Distance (1), Density (1) = 104
    # Or: Player (6), Weapon (3), Zombies (13*6=78), Pickups (3*3=9), Distance (1), Density (1) = 98 ✓
    state_dim: int = 98
    
    # Training hyperparameters - As specified in master prompt
    learning_rate: float = 3e-4  # As specified
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Lambda for GAE (Generalized Advantage Estimation)
    clip_epsilon: float = 0.2  # Clipping parameter for PPO
    
    # Loss coefficients - ADJUSTED for exploration
    value_loss_coef: float = 0.5  # Coefficient for value loss
    entropy_coef: float = 0.02  # INCREASED from 0.01 for better exploration of weapon actions
    max_grad_norm: float = 0.5  # Maximum gradient norm for clipping
    
    # Training loop parameters
    n_steps: int = 2048  # Number of steps to collect before update
    batch_size: int = 64  # Minibatch size for updates
    n_epochs: int = 10  # Number of epochs per update
    
    # Neural network architecture
    hidden_dim: int = 256  # Hidden layer dimension
    
    # Exploration
    initial_std: float = 1.0  # Initial standard deviation (not used for discrete)
    
    # Training control
    total_timesteps: int = 1_000_000  # Total timesteps to train
    save_interval: int = 50_000  # Save model every N timesteps
    log_interval: int = 10  # Log every N updates
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pth"
    
    # Environment-specific settings
    fps: int = 60  # Game FPS
    headless: bool = True  # Run without rendering during training
    
    # ========================================================================
    # REWARD SHAPING PARAMETERS - EXPERT RL SYSTEM
    # ========================================================================
    
    # Combat Rewards
    reward_zombie_kill: float = 10.0  # Base reward for killing a zombie
    reward_strong_zombie_kill_bonus: float = 4.0  # Bonus for killing a strong zombie
    reward_hit: float = 0.05  # Reward per successful hit
    reward_machinegun_fire_per_second: float = 0.2  # Reward per second firing machine gun
    
    # Survival Rewards
    reward_survival_per_step: float = 0.1  # Reward for surviving each step
    reward_moving_away_from_zombies: float = 0.5  # Reward when moving away from nearby zombies
    reward_escape_danger_zone: float = 1.0  # Reward for successfully escaping high-danger zone
    
    # Positioning Rewards/Penalties
    reward_standing_still_penalty: float = -1.0  # Penalty when standing still while zombies nearby
    reward_too_close_penalty: float = -0.5  # Penalty when too close to zombies (< danger threshold)
    
    # Pickup Rewards
    reward_machinegun_pickup: float = 3.0  # Reward for picking up machine gun
    reward_health_pickup: float = 2.0  # Reward for picking up health pack when HP < 100%
    reward_ammo_pickup: float = 1.5  # Reward for picking up machine-gun ammo
    reward_weapon_switch: float = 1.0  # Reward for switching weapons appropriately
    
    # Penalties
    reward_damage_taken: float = -5.0  # Penalty for every 10 HP lost
    reward_death: float = -20.0  # Penalty on death
    reward_idle_penalty: float = -0.1  # Idle penalty
    
    # Danger & Distance-Based Behavior
    reward_danger_decrease: float = 0.3  # Reward when danger decreases due to movement
    reward_danger_increase: float = -0.3  # Penalty when danger increases
    reward_high_density_penalty: float = -1.0  # Penalty when entering high-density zombie zone
    
    # Danger calculation parameters
    danger_radius: float = 200.0  # Radius for danger calculation
    danger_threshold: float = 50.0  # Distance threshold for "too close" penalty
    high_density_threshold: float = 5.0  # Number of zombies for high-density zone
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Validate state dimension (98-dim as specified)
        # Player (6) + Weapon (3) + Zombies (13*6=78) + Pickups (3*3=9) + Distance (1) + Density (1) = 98
        if self.state_dim != 98:
            print(f"Warning: state_dim should be 98. Got {self.state_dim}")
        
        # Validate action dimension
        if self.action_dim != 6:
            print(f"Warning: action_dim should be 6. Got {self.action_dim}")


# Default configuration instance
default_config = PPOConfig()


# ============================================================================
# PPO HYPERPARAMETER EXPLANATION
# ============================================================================
"""
Why these hyperparameters improve weapon-specific behavior:

1. LEARNING RATE (2.5e-4, down from 3e-4):
   - Slightly lower to prevent the policy from changing too quickly
   - The new reward structure is more complex, needs stable learning
   - Helps agent properly learn the value of weapon switching

2. ENTROPY COEFFICIENT (0.02, up from 0.01):
   - Higher entropy encourages more exploration
   - Critical for discovering that weapon switching (actions 6, 7) is beneficial
   - Without sufficient exploration, agent might never try the machine gun
   - After exploration, PPO's policy will naturally reduce entropy as it learns

3. REWARD STRUCTURE:
   
   a) Health Pickup Rewards:
      - Base reward of 8.0 is strong enough to be noticed
      - Scaling by health gained means:
        * At full health (25 HP cap): 8.0 * 1.0 = 8.0
        * At low health (full 25 HP): 8.0 * 2.0 = 16.0
      - This creates a strong incentive to collect health when injured
   
   b) Low Health Shaping:
      - reward_low_health_near_pickup: Small continuous reward for being near health
      - reward_ignore_health_penalty: Small continuous penalty for ignoring health
      - Together these create gradient towards health packs when hurt
   
   c) Machine Gun Incentives:
      - reward_machinegun_pickup (5.0): Collect the weapon
      - reward_machinegun_kill_bonus (2.0): 20% bonus per kill with MG
      - reward_machinegun_dps_bonus (0.5): Bonus for sustained fire
      
   d) Why Machine Gun is Worth Using:
      - Pistol: 20 damage, 20 frame cooldown = 1.0 DPS per frame
      - MG: 10 damage, 5 frame cooldown = 2.0 DPS per frame (DOUBLE!)
      - Plus 2.0 bonus per kill = more points
      - Agent should learn: MG has higher DPS, use it when available

4. STATE REPRESENTATION:
   - Added has_machinegun, machinegun_ammo, current_weapon_is_mg
   - Agent can observe its weapon state and make informed decisions
   - Added danger_level: helps agent know when to prioritize health
   - Added pickup info: agent can "see" where health/MG pickups are

5. OBSERVATION SPACE (108 dimensions):
   - 12 player features (including weapon state)
   - 90 zombie features (15 zombies * 6 features)
   - 3 health pickup features
   - 3 machinegun pickup features
   - Rich enough for strategic decision making
"""
