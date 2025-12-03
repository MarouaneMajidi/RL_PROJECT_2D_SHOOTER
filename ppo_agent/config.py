"""
PPO Configuration File

Contains all hyperparameters for the PPO algorithm.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PPOConfig:
    """Configuration class for PPO hyperparameters."""
    
    # Environment settings
    max_zombies: int = 15  # Maximum number of zombies to track in state
    state_dim: int = 77  # State dimension (will be computed in env)
    action_dim: int = 6  # 6 discrete actions
    
    # Training hyperparameters
    learning_rate: float = 3e-4  # Learning rate for optimizer
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Lambda for GAE (Generalized Advantage Estimation)
    clip_epsilon: float = 0.2  # Clipping parameter for PPO
    
    # Loss coefficients
    value_loss_coef: float = 0.5  # Coefficient for value loss
    entropy_coef: float = 0.01  # Coefficient for entropy bonus
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
    
    # Reward shaping parameters
    reward_zombie_kill: float = 10.0
    reward_survival_per_step: float = 0.1
    reward_damage_taken: float = -5.0
    reward_death: float = -20.0
    reward_idle_penalty: float = -0.01
    reward_distance_to_zombie: float = 0.01  # Small reward for getting closer to zombies
    reward_health_pickup: float = 5.0  # Reward for collecting health pickup
    reward_machinegun_pickup: float = 3.0  # Reward for collecting machine gun pickup
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Compute actual state dimension
        # Player: [x, y, vel_x, vel_y, health, shoot_cooldown, angle] = 7
        # Per zombie: [x, y, health, type, distance, angle_to_player] = 6
        # Total: 7 + (max_zombies * 6) + 1 (zombie count) = 7 + 1 + 60 = 68 for 10 zombies
        self.state_dim = 8 + (self.max_zombies * 6)


# Default configuration instance
default_config = PPOConfig()
