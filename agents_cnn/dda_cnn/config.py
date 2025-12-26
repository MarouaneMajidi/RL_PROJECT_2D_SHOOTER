"""
Configuration for CNN-based DDA Agent

PPO hyperparameters and image preprocessing settings for CNN-based DDA training.
"""

from dataclasses import dataclass
import torch


@dataclass
class CNNDDAConfig:
    """
    Configuration class for CNN-based DDA Agent.
    
    Uses raw RGB images as input instead of manually extracted features.
    """
    
    # Image preprocessing settings
    RESIZE_WIDTH: int = 128
    RESIZE_HEIGHT: int = 96
    NUM_STACKED_FRAMES: int = 4
    COLOR_CHANNELS: int = 3  # RGB (NOT grayscale)
    
    # Input shape: (4 frames, 96 height, 128 width, 3 RGB channels)
    # For PyTorch: (batch, channels, height, width) = (batch, 12, 96, 128)
    # First conv layer sees 12 input channels (4 frames × 3 RGB channels)
    CNN_INPUT_CHANNELS: int = NUM_STACKED_FRAMES * COLOR_CHANNELS  # 12
    
    # Action space (same as feature-based DDA agent)
    action_dim: int = 5  # Discrete actions: 0=do nothing, 1-2=easier, 3-4=harder
    
    # CNN Architecture hyperparameters
    cnn_hidden_dim: int = 64  # Hidden dimension for CNN features
    feature_dim: int = 128  # Dimension after CNN feature extraction (smaller for DDA)
    
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
    n_epochs: int = 4  # Reduced from 10 to prevent overfitting to single rollouts
    
    # DDA-specific settings
    action_interval: int = 300  # Frames between DDA actions (5 seconds at 60 FPS)
    episode_length: int = 3600  # Episode length in frames (60 seconds at 60 FPS)
    
    # Training control
    total_timesteps: int = 500_000
    save_interval: int = 50_000
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints_cnn/dda_cnn"
    best_model_path: str = "checkpoints_cnn/dda_cnn/best_model.pth"
    
    # Environment settings
    fps: int = 60
    headless: bool = True
