"""
Rollout Buffer for CNN-based PPO Agents

Stores image-based trajectories and computes advantages using GAE.
States are images with shape (12, 96, 128) instead of 1D vectors.
"""

import torch
import numpy as np
from typing import Generator, Tuple


class CNNRolloutBuffer:
    """
    Buffer for storing image-based trajectories and computing advantages with GAE.
    
    States are stored as images: (buffer_size, 12, 96, 128)
    where 12 = 4 frames × 3 RGB channels
    """
    
    def __init__(self, buffer_size: int, state_shape: Tuple[int, ...], action_dim: int, device: str = "cpu"):
        """
        Initialize the rollout buffer.
        
        Args:
            buffer_size: Maximum size of the buffer
            state_shape: Shape of state (channels, height, width) = (12, 96, 128)
            action_dim: Dimension of the action space
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape  # (12, 96, 128)
        self.action_dim = action_dim
        self.device = device
        
        # Initialize storage for image states
        # Shape: (buffer_size, 12, 96, 128)
        self.states = torch.zeros((buffer_size, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size,), dtype=torch.long)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32)
        
        # For GAE computation
        self.advantages = torch.zeros((buffer_size,), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size,), dtype=torch.float32)
        
        # Position in buffer
        self.pos = 0
        self.full = False
    
    def add(self, state: np.ndarray, action: int, reward: float, value: float, 
            log_prob: float, done: bool):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state image (12, 96, 128)
            action: Action taken
            reward: Reward received
            value: Value estimate from critic
            log_prob: Log probability of the action
            done: Whether episode is done
        """
        self.states[self.pos] = torch.from_numpy(state).float()
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, 
                                       gae_lambda: float = 0.95):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for the last state (bootstrap value)
            gamma: Discount factor
            gae_lambda: Lambda parameter for GAE
        """
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE: A_t = δ_t + γ * λ * A_{t+1}
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Returns are advantages + values
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate random mini-batches from the buffer.
        
        Args:
            batch_size: Size of mini-batches
        
        Yields:
            Tuple of (states, actions, old_log_probs, returns, advantages, old_values)
            states shape: (batch_size, 12, 96, 128)
        """
        # Normalize advantages
        advantages_normalized = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
        # Generate random indices
        indices = torch.randperm(self.buffer_size)
        
        # Yield mini-batches
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                self.states[batch_indices].to(self.device),
                self.actions[batch_indices].to(self.device),
                self.log_probs[batch_indices].to(self.device),
                self.returns[batch_indices].to(self.device),
                advantages_normalized[batch_indices].to(self.device),
                self.values[batch_indices].to(self.device)
            )
    
    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        
        # Clear tensors
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
        self.advantages.zero_()
        self.returns.zero_()
    
    def size(self) -> int:
        """Return the current size of the buffer."""
        return self.buffer_size if self.full else self.pos
