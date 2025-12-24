"""
Environment Wrapper for CNN-based Agents

Wraps the existing ZombieShooterEnv to provide image-based observations
without modifying the original environment.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from .image_preprocessor import ImagePreprocessor

# Import existing environment (READ ONLY - no modifications)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.ppo_agent.env import ZombieShooterEnv
from agents.ppo_agent.config import PPOConfig


class CNNEnvWrapper:
    """
    Wrapper around ZombieShooterEnv that provides image-based observations.
    
    This wrapper:
    - Captures screenshots from the game
    - Preprocesses images (resize, stack frames)
    - Maintains the same action/reward interface as original env
    - Does NOT modify the original ZombieShooterEnv class
    """
    
    def __init__(self, config: PPOConfig, headless: bool = True):
        """
        Initialize the CNN environment wrapper.
        
        Args:
            config: PPOConfig from original agent (for environment setup)
            headless: Whether to run without displaying window (for faster training)
                     NOTE: Even if headless=True, we MUST render to capture images for CNN!
                     
        IMPORTANT: This wrapper captures images from ZombieShooterEnv.render()
        Currently, ZombieShooterEnv uses simple circles (not sprite assets).
        To train on the real game with assets, you would need to modify ZombieShooterEnv
        to use sprite rendering like main.py does.
        """
        # CRITICAL: For CNN, we MUST render to capture images!
        # ZombieShooterEnv.render() returns early if headless=True
        # So we MUST use headless=False so render() actually draws to screen surface
        # We can hide the window display using SDL_VIDEODRIVER=dummy if needed
        import os
        
        # If headless, set dummy video driver to hide window (but still allow rendering)
        if headless and 'SDL_VIDEODRIVER' not in os.environ:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        # Create environment with headless=False so render() actually works
        # This is REQUIRED for CNN to capture images from screen surface
        self.env = ZombieShooterEnv(config, headless=False)
        
        # Initialize image preprocessor
        # Get arena offsets from environment
        self.preprocessor = ImagePreprocessor()
        # Update offsets to match environment
        self.preprocessor.ARENA_X_OFFSET = self.env.ARENA_X_OFFSET
        self.preprocessor.ARENA_Y_OFFSET = self.env.ARENA_Y_OFFSET
        
        # Store original headless state
        self._headless = headless
        
        # Observation shape: (4, 96, 128, 3) for stacked RGB frames
        # For PyTorch, we'll reshape to (12, 96, 128) channels-first format
        self.observation_shape = (12, 96, 128)  # (channels, height, width)
        
        # Action space (same as original)
        self.action_space_shape = config.action_space_shape
        self.action_dim = config.action_dim
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial observation.
        
        Returns:
            Initial stacked frames as numpy array (12, 96, 128)
            Format: (channels, height, width) where channels = 4 frames × 3 RGB
        """
        # Reset underlying environment
        _ = self.env.reset()
        
        # Reset frame buffer
        self.preprocessor.reset()
        
        # Render to get initial frame
        self.env.render()
        
        # Add initial frame (will be repeated 4 times)
        self.preprocessor.add_frame(self.env.screen)
        
        # Get stacked frames and convert to channels-first format for PyTorch
        stacked = self.preprocessor.get_stacked_frames()  # (4, 96, 128, 3)
        
        # Reshape to (12, 96, 128) - channels first format
        # Stack RGB channels from each frame: frame0_r, frame0_g, frame0_b, frame1_r, ...
        observation = stacked.reshape(12, 96, 128)
        
        return observation.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Combined action value (movement * 2 + shoot)
        
        Returns:
            observation: Stacked frames (12, 96, 128)
            reward: Reward from the environment
            done: Whether episode is done
            info: Info dictionary from environment
        """
        # Step underlying environment
        _, reward, done, info = self.env.step(action)
        
        # Render to get updated frame
        self.env.render()
        
        # Add new frame to buffer
        self.preprocessor.add_frame(self.env.screen)
        
        # Get stacked frames and convert to channels-first format
        stacked = self.preprocessor.get_stacked_frames()  # (4, 96, 128, 3)
        observation = stacked.reshape(12, 96, 128)  # (12, 96, 128)
        
        return observation.astype(np.float32), reward, done, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def render(self):
        """Render the environment."""
        self.env.render()
