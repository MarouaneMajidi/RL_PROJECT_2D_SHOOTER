"""
CNN Environment Wrapper for DDA Agent

Wraps the existing DDAEnvironment to provide image-based observations
without modifying the original environment.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional

# Import image preprocessor from player_cnn (shared implementation)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_cnn.image_preprocessor import ImagePreprocessor


class CNNDDAEnvWrapper:
    """
    Wrapper around DDAEnvironment that provides image-based observations.
    
    This wrapper:
    - Captures screenshots from the underlying game environment
    - Preprocesses images (resize, stack frames)
    - Maintains the same action/reward interface as original DDAEnvironment
    - Does NOT modify the original DDAEnvironment class
    """
    
    def __init__(self, dda_env):
        """
        Initialize the CNN DDA environment wrapper.
        
        Args:
            dda_env: DDAEnvironment instance (feature-based)
        """
        # Store underlying DDA environment
        self.dda_env = dda_env
        
        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Update offsets to match the underlying game environment
        if hasattr(dda_env, 'game_env'):
            self.preprocessor.ARENA_X_OFFSET = dda_env.game_env.ARENA_X_OFFSET
            self.preprocessor.ARENA_Y_OFFSET = dda_env.game_env.ARENA_Y_OFFSET
        
        # Observation shape: (12, 96, 128) for stacked RGB frames
        self.observation_shape = (12, 96, 128)  # (channels, height, width)
        
        # Action space (same as original DDA)
        self.action_dim = dda_env.difficulty_manager.adjustment_amounts.__len__() if hasattr(dda_env.difficulty_manager, 'adjustment_amounts') else 5
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial observation.
        
        Returns:
            Initial stacked frames as numpy array (12, 96, 128)
        """
        # Reset underlying DDA environment
        _ = self.dda_env.reset()
        
        # Reset frame buffer
        self.preprocessor.reset()
        
        # Render to get initial frame
        if hasattr(self.dda_env.game_env, 'render'):
            self.dda_env.game_env.render()
        
        # Capture image from underlying game environment
        if hasattr(self.dda_env.game_env, 'screen'):
            self.preprocessor.add_frame(self.dda_env.game_env.screen)
        
        # Get stacked frames and convert to channels-first format
        stacked = self.preprocessor.get_stacked_frames()  # (4, 96, 128, 3)
        observation = stacked.reshape(12, 96, 128)  # (12, 96, 128)
        
        return observation.astype(np.float32)
    
    def step(self, dda_action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the DDA environment.
        
        Args:
            dda_action: DDA action (0-4)
        
        Returns:
            observation: Stacked frames (12, 96, 128)
            reward: Reward from DDA environment
            done: Whether episode is done
            info: Info dictionary from DDA environment
        """
        # Step underlying DDA environment
        _, reward, done, info = self.dda_env.step(dda_action)
        
        # Render to get updated frame (if game_env has render)
        if hasattr(self.dda_env.game_env, 'render'):
            self.dda_env.game_env.render()
        
        # Capture image from underlying game environment
        if hasattr(self.dda_env.game_env, 'screen'):
            self.preprocessor.add_frame(self.dda_env.game_env.screen)
        
        # Get stacked frames and convert to channels-first format
        stacked = self.preprocessor.get_stacked_frames()  # (4, 96, 128, 3)
        observation = stacked.reshape(12, 96, 128)  # (12, 96, 128)
        
        return observation.astype(np.float32), reward, done, info
    
    def close(self):
        """Close the environment."""
        if hasattr(self.dda_env, 'close'):
            self.dda_env.close()
    
    def set_player_agent(self, player_agent):
        """Set the player agent in the underlying DDA environment."""
        if hasattr(self.dda_env, 'set_player_agent'):
            self.dda_env.set_player_agent(player_agent)
        elif hasattr(self.dda_env, 'player_agent'):
            self.dda_env.player_agent = player_agent
