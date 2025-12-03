"""
PPO Configuration File

Contains all hyperparameters for the PPO algorithm.

UPDATED:
- Extended action space (8 actions) for weapon switching
- Extended state space for pickup detection, weapon state, danger level
- New reward shaping parameters for health pickups and machine gun usage
- Optimized hyperparameters for weapon-based learning
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
    
    # Action space: 0-3 movement, 4 shoot, 5 idle, 6 switch pistol, 7 switch machinegun
    action_dim: int = 8  # Extended from 6 to 8 for weapon switching
    
    # State space calculation:
    # Player: 12 values [x, y, vel_x, vel_y, health, shoot_cooldown, angle, zombie_count,
    #                    has_machinegun, machinegun_ammo, current_weapon_is_mg, danger_level]
    # Zombies: 6 * max_zombies = 90 values
    # Health pickup: 3 values [distance, angle, exists]
    # Machinegun pickup: 3 values [distance, angle, exists]
    # Total: 12 + 90 + 3 + 3 = 108
    state_dim: int = 108
    
    # Training hyperparameters - OPTIMIZED for weapon learning
    learning_rate: float = 2.5e-4  # Slightly lower for stability with new reward structure
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
    # REWARD SHAPING PARAMETERS
    # ========================================================================
    
    # Core rewards
    reward_zombie_kill: float = 10.0  # Base reward for killing a zombie
    reward_survival_per_step: float = 0.1  # Reward for surviving each step
    reward_damage_taken: float = -5.0  # Penalty per 10 HP of damage taken
    reward_death: float = -20.0  # Penalty for dying
    reward_idle_penalty: float = -0.01  # Small penalty for staying idle
    reward_distance_to_zombie: float = 0.01  # Small reward for approaching zombies
    
    # Health pickup rewards - ENHANCED
    reward_health_pickup: float = 8.0  # Base reward for collecting health pickup
    #   NOTE: Actual reward is scaled by health gained (low health = more reward)
    #   Formula: reward_health_pickup * (1.0 + health_gained/25)
    
    reward_low_health_near_pickup: float = 0.1  # Reward per step when low health and near health pickup
    #   Encourages agent to move towards health when injured
    
    reward_ignore_health_penalty: float = -0.05  # Penalty for ignoring health pickup when low on health
    #   Applied when health < 50 and health pickup exists but agent is not near it
    
    # Machine gun rewards - NEW
    reward_machinegun_pickup: float = 7.0  # Reward for collecting machine gun pickup
    
    reward_machinegun_kill_bonus: float = 5.0  # Bonus reward for kills with machine gun
    #   Total kill reward with MG = reward_zombie_kill + reward_machinegun_kill_bonus = 15.0
    #   This incentivizes using the MG when available
    
    reward_machinegun_dps_bonus: float = 0.5  # Bonus for sustained machine gun fire
    #   Applied when 5+ MG shots are fired within 1 second
    #   Encourages continuous fire rather than single shots
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Validate state dimension matches expected calculation
        expected_state_dim = 12 + (self.max_zombies * 6) + 3 + 3
        if self.state_dim != expected_state_dim:
            print(f"Warning: state_dim mismatch. Expected {expected_state_dim}, got {self.state_dim}")
            self.state_dim = expected_state_dim
        
        # Validate action dimension
        if self.action_dim != 8:
            print(f"Warning: action_dim should be 8 for weapon switching. Got {self.action_dim}")


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
