"""
Image Preprocessing Module for CNN-based Agents

Captures game screenshots, resizes to 128x96 RGB, and stacks frames for temporal information.
Maintains RGB color channels - does NOT convert to grayscale.
"""

import numpy as np
import pygame
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Preprocesses game screenshots for CNN input.
    
    - Captures 800×600 arena region
    - Resizes to 128×96 RGB (maintains 4:3 aspect ratio)
    - Stacks 4 consecutive frames for temporal information
    - Normalizes pixel values to [0, 1]
    - Maintains RGB color channels (NO grayscale conversion)
    """
    
    # Configuration
    RESIZE_WIDTH = 128
    RESIZE_HEIGHT = 96
    NUM_STACKED_FRAMES = 4
    COLOR_CHANNELS = 3  # RGB
    GRAYSCALE = False  # Always keep RGB
    
    # Original game dimensions (will be updated by env_wrapper)
    ARENA_WIDTH = 800
    ARENA_HEIGHT = 600
    ARENA_X_OFFSET = 100  # Default: (1000 - 800) / 2, updated by env_wrapper
    ARENA_Y_OFFSET = 100  # Default: (800 - 600) / 2, updated by env_wrapper
    
    def __init__(self):
        """Initialize the image preprocessor."""
        # Frame buffer for stacking
        self.frame_buffer: list = []
        self.initialized = False
        
        # Input shape after preprocessing: (4, 96, 128, 3)
        # 4 frames × 96 height × 128 width × 3 RGB channels
        self.output_shape = (self.NUM_STACKED_FRAMES, self.RESIZE_HEIGHT, self.RESIZE_WIDTH, self.COLOR_CHANNELS)
    
    def capture_arena(self, screen: pygame.Surface) -> np.ndarray:
        """
        Capture the arena region from the game screen.
        
        Args:
            screen: Pygame surface containing the full game screen
            
        Returns:
            numpy array of shape (600, 800, 3) containing RGB arena pixels
        """
        # Extract arena region (800×600)
        arena_region = pygame.Surface((self.ARENA_WIDTH, self.ARENA_HEIGHT))
        arena_region.blit(screen, (0, 0), 
                         (self.ARENA_X_OFFSET, self.ARENA_Y_OFFSET, 
                          self.ARENA_WIDTH, self.ARENA_HEIGHT))
        
        # Convert to numpy array (RGB format)
        # pygame uses (width, height) but numpy uses (height, width)
        array = pygame.surfarray.array3d(arena_region)
        # Transpose from (width, height, channels) to (height, width, channels)
        array = np.transpose(array, (1, 0, 2))
        
        return array.astype(np.float32)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image from 800×600 to 128×96 RGB.
        
        Uses strided sampling for fast downsampling.
        
        Args:
            image: numpy array of shape (600, 800, 3) in RGB format
            
        Returns:
            numpy array of shape (96, 128, 3) in RGB format
        """
        height, width = image.shape[:2]
        target_height, target_width = self.RESIZE_HEIGHT, self.RESIZE_WIDTH
        
        # Calculate stride for downsampling
        stride_y = height / target_height
        stride_x = width / target_width
        
        # Create indices for downsampling
        y_indices = (np.arange(target_height) * stride_y).astype(np.int32)
        x_indices = (np.arange(target_width) * stride_x).astype(np.int32)
        
        # Clip indices to valid range
        y_indices = np.clip(y_indices, 0, height - 1)
        x_indices = np.clip(x_indices, 0, width - 1)
        
        # Sample pixels using advanced indexing (fast)
        resized = image[np.ix_(y_indices, x_indices)]
        
        return resized.astype(np.float32)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values from [0, 255] to [0, 1].
        
        Args:
            image: numpy array with pixel values in [0, 255]
            
        Returns:
            numpy array with pixel values in [0, 1]
        """
        return image / 255.0
    
    def process_frame(self, screen: pygame.Surface) -> np.ndarray:
        """
        Process a single frame: capture → resize → normalize.
        
        Args:
            screen: Pygame surface containing the game screen
            
        Returns:
            numpy array of shape (96, 128, 3) with normalized RGB values
        """
        # Capture arena
        arena = self.capture_arena(screen)
        
        # Resize to 128×96
        resized = self.resize_image(arena)
        
        # Normalize to [0, 1]
        normalized = self.normalize_image(resized)
        
        return normalized
    
    def add_frame(self, screen: pygame.Surface) -> None:
        """
        Add a new frame to the buffer and maintain stack of 4 frames.
        
        Args:
            screen: Pygame surface containing the game screen
        """
        frame = self.process_frame(screen)
        
        if not self.initialized:
            # Initialize buffer with first frame repeated 4 times
            self.frame_buffer = [frame.copy() for _ in range(self.NUM_STACKED_FRAMES)]
            self.initialized = True
        else:
            # Add new frame and remove oldest
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.NUM_STACKED_FRAMES:
                self.frame_buffer.pop(0)
    
    def get_stacked_frames(self) -> np.ndarray:
        """
        Get the current stack of 4 frames.
        
        Returns:
            numpy array of shape (4, 96, 128, 3) for CNN input
            Format: (num_frames, height, width, channels)
        """
        if not self.initialized:
            raise RuntimeError("Frame buffer not initialized. Call add_frame() first.")
        
        # Stack frames: (4, 96, 128, 3)
        stacked = np.stack(self.frame_buffer, axis=0)
        return stacked.astype(np.float32)
    
    def reset(self) -> None:
        """Reset the frame buffer (call at episode start)."""
        self.frame_buffer = []
        self.initialized = False
    
    def get_state_shape(self) -> Tuple[int, int, int, int]:
        """
        Get the shape of the preprocessed state.
        
        Returns:
            Tuple of (num_frames, height, width, channels)
        """
        return self.output_shape
