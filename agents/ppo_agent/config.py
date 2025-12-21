"""
PPO Configuration File - EXPERT RL SYSTEM REDESIGN

COMPREHENSIVE REDESIGN FOR OPTIMAL AGENT BEHAVIOR:
- Multi-discrete action space: Movement (5) + Shoot (2) = simultaneous actions
- Enhanced observation space with directional pickup information
- Dense directional reward shaping for pickup seeking and combat
- Improved environment mechanics for reliable pickup collection
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class PPOConfig:
    """
    Configuration class for PPO hyperparameters.
    
    REDESIGNED FOR OPTIMAL BEHAVIOR:
    1. Multi-discrete action space: [movement_action, shoot_action]
       - Movement: 5 actions (up, down, left, right, idle)
       - Shoot: 2 actions (shoot, don't shoot)
       - Allows simultaneous movement and shooting
    
    2. Enhanced state space with directional pickup information
    
    3. Dense directional reward shaping:
       - Rewards for moving TOWARD pickups (not just being near)
       - Rewards for shooting while moving
       - Penalties for running away without shooting
       - Progressive rewards for pickup proximity
    """
    
    # Environment settings
    max_zombies: int = 15  # Maximum number of zombies to track in state
    
    # Multi-discrete action space: [movement_action, shoot_action]
    # Movement: 0=Up, 1=Down, 2=Left, 3=Right, 4=Idle
    # Shoot: 0=Don't shoot, 1=Shoot
    action_space_shape: Tuple[int, ...] = (5, 2)  # Multi-discrete: [movement, shoot]
    action_dim: int = 5 * 2  # Total combinations (for compatibility, but we use multi-discrete)
    
    # Enhanced state space calculation (129-dim):
    # Player: 6 values [x, y, health, shoot_cooldown, angle, zombie_count]
    # Weapon: 3 values [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg]
    # Zombies: 7 * 13 = 91 values [x, y, health, type, distance, angle_to_player, in_bounds] per zombie
    #   * in_bounds flag helps agent distinguish valid targets from off-screen zombies
    # Pickups (enhanced with directional info):
    #   Machinegun: [distance, angle, exists, dx_normalized, dy_normalized, in_range] = 6
    #   Health: [distance, angle, exists, dx_normalized, dy_normalized, in_range] = 6
    #   Ammo: [distance, angle, exists, dx_normalized, dy_normalized, in_range] = 6
    #   Total pickups: 18 values
    # Distance to nearest IN-BOUNDS zombie: 1 value
    # Zombie density in radius (in-bounds only): 1 value
    # Movement direction (last action): 4 values [up, down, left, right]
    # Pickup urgency signals: 3 values [health_urgency, machinegun_urgency, ammo_urgency]
    # Zombie counts: 2 values [in_bounds_count_normalized, total_count_normalized]
    # Total: 6 + 3 + 91 + 18 + 1 + 1 + 4 + 3 + 2 = 129
    state_dim: int = 129
    
    # Training hyperparameters - Optimized for complex behavior learning
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Lambda for GAE
    clip_epsilon: float = 0.2  # Clipping parameter for PPO
    
    # Loss coefficients - Adjusted for better exploration
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.03  # Increased for better exploration of multi-discrete space
    max_grad_norm: float = 0.5
    
    # Training loop parameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    
    # Neural network architecture
    hidden_dim: int = 256
    
    # Exploration
    initial_std: float = 1.0
    
    # Training control
    total_timesteps: int = 1_000_000
    save_interval: int = 50_000
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pth"
    
    # Environment-specific settings
    fps: int = 60
    headless: bool = True
    
    # ========================================================================
    # ENHANCED REWARD SHAPING - DIRECTIONAL AND DENSE
    # ========================================================================
    
    # Combat Rewards
    reward_zombie_kill: float = 10.0
    reward_strong_zombie_kill_bonus: float = 4.0
    reward_hit: float = 0.1  # Increased for better signal
    reward_machinegun_fire_per_second: float = 0.3  # Increased
    
    # Survival Rewards
    reward_survival_per_step: float = 0.1
    reward_moving_away_from_zombies: float = 0.5
    reward_escape_danger_zone: float = 1.0
    
    # Positioning Rewards/Penalties
    reward_standing_still_penalty: float = -1.5  # Increased penalty
    reward_too_close_penalty: float = -0.5
    
    # Pickup Rewards - ENHANCED WITH DIRECTIONAL REWARDS
    reward_machinegun_pickup: float = 5.0  # Increased
    reward_health_pickup: float = 3.0  # Increased
    reward_ammo_pickup: float = 2.0  # Increased
    reward_weapon_switch: float = 1.0
    
    # NEW: Directional pickup rewards (dense shaping, urgency-weighted)
    reward_moving_toward_machinegun: float = 0.3  # Increased - machinegun is critical
    reward_moving_toward_health: float = 0.2  # Increased - health extends survival
    reward_moving_toward_ammo: float = 0.15  # Increased - ammo maintains combat capability
    reward_pickup_proximity: float = 0.1  # Increased - being near pickups is good
    
    # NEW: Strategic pickup collection rewards (context-aware)
    reward_strategic_health_pickup: float = 2.0  # Bonus for collecting health when low
    reward_strategic_ammo_pickup: float = 1.5  # Bonus for collecting ammo when low on ammo
    reward_strategic_machinegun_pickup: float = 3.0  # Bonus for collecting machinegun (always strategic)
    
    # NEW: Combat behavior rewards
    reward_shooting_while_moving: float = 0.3  # Reward for shooting while moving
    reward_shooting_at_enemies: float = 0.2  # Reward when shooting with enemies in range
    penalty_running_away_without_shooting: float = -0.5  # Penalty for fleeing without fighting
    penalty_shooting_at_out_of_bounds: float = -0.3  # Penalty for shooting when no in-bounds zombies
    
    # Penalties
    reward_damage_taken: float = -5.0
    reward_death: float = -20.0
    reward_idle_penalty: float = -0.1
    
    # Danger & Distance-Based Behavior
    reward_danger_decrease: float = 0.3
    reward_danger_increase: float = -0.3
    reward_high_density_penalty: float = -1.0
    
    # Danger calculation parameters
    danger_radius: float = 200.0
    danger_threshold: float = 50.0
    high_density_threshold: float = 5.0
    
    # Pickup collection parameters
    pickup_collection_radius: float = 40.0  # Increased from 25 for easier collection
    pickup_detection_range: float = 300.0  # Range for "in_range" flag
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Validate state dimension
        if self.state_dim != 129:
            print(f"Warning: state_dim should be 129. Got {self.state_dim}")
        
        # Validate action space
        if len(self.action_space_shape) != 2:
            print(f"Warning: action_space_shape should be (5, 2). Got {self.action_space_shape}")


# Default configuration instance
default_config = PPOConfig()
