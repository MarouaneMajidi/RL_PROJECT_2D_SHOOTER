"""
DDA Agent Configuration

Configuration for Dynamic Difficulty Adjustment agent.
"""

from dataclasses import dataclass
import torch


@dataclass
class DDAConfig:
    """
    Configuration class for DDA agent hyperparameters.
    """
    
    # Environment settings
    state_dim: int = 9  # DDA state dimension
    action_dim: int = 5  # Discrete actions: 0=do nothing, 1-2=easier, 3-4=harder
    action_interval: int = 300  # Frames between DDA actions (5 seconds at 60 FPS)
    episode_length: int = 3600  # Episode length in frames (60 seconds at 60 FPS)
    
    # Training hyperparameters
    learning_rate: float = 5e-4  # Increased from 3e-4 to encourage faster policy updates
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Lambda for GAE
    clip_epsilon: float = 0.2  # Clipping parameter for PPO
    
    # Loss coefficients
    value_loss_coef: float = 0.25  # Reduced from 0.5 to stabilize value function and reduce value loss spikes
    entropy_coef: float = 0.15  # Increased from 0.01 to prevent policy collapse
    max_grad_norm: float = 0.5
    
    # Training loop parameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 4  # Reduced from 10 to prevent overfitting to old data
    
    # Neural network architecture
    hidden_dim: int = 128
    
    # Training control
    total_timesteps: int = 500_000
    save_interval: int = 50_000
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints/dda"
    best_model_path: str = "checkpoints/dda/best_dda_model.pth"

