"""
Configuration for CNN-based Player Agent

PPO hyperparameters and image preprocessing settings for CNN-based training.
"""

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class CNNPlayerConfig:
    """
    Configuration class for CNN-based Player Agent.
    
    Uses raw RGB images as input instead of manually extracted features.
    """
    
    # Image preprocessing settings
    RESIZE_WIDTH: int = 128
    RESIZE_HEIGHT: int = 96
    NUM_STACKED_FRAMES: int = 4
    COLOR_CHANNELS: int = 3  # RGB (NOT grayscale)
    
    # Input shape: (4 frames, 96 height, 128 width, 3 RGB channels)
    # For PyTorch: (batch, channels, frames*3, height, width) or flattened
    # We'll reshape to (batch, frames*3, height, width) = (batch, 12, 96, 128)
    # First conv layer sees 12 input channels (4 frames × 3 RGB channels)
    CNN_INPUT_CHANNELS: int = NUM_STACKED_FRAMES * COLOR_CHANNELS  # 12
    
    # Action space (same as feature-based agent)
    # Multi-discrete: Movement (5) × Shoot (2)
    action_space_shape: Tuple[int, ...] = (5, 2)
    action_dim: int = 5 * 2  # Total combinations
    
    # CNN Architecture hyperparameters
    cnn_hidden_dim: int = 64  # Hidden dimension for CNN features
    feature_dim: int = 256  # Dimension after CNN feature extraction
    
    # Training hyperparameters
    learning_rate: float = 3e-4  # Reduced from 5e-4 for better stability (prevents collapse)
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Lambda for GAE
    clip_epsilon: float = 0.2  # Clipping parameter for PPO
    
    # Loss coefficients
    value_loss_coef: float = 0.5  # Restored to 0.5 for better value function stability
    entropy_coef: float = 0.15  # Increased from 0.01 to prevent premature policy collapse
    max_grad_norm: float = 0.5
    
    # Training loop parameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 4  # Reduced from 10 to prevent overfitting to old data
    
    # Training control
    total_timesteps: int = 1_000_000
    save_interval: int = 50_000
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints_cnn/player_cnn"
    best_model_path: str = "checkpoints_cnn/player_cnn/best_model.pth"
    
    # Environment settings
    fps: int = 60
    headless: bool = True
