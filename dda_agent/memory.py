"""
Rollout Buffer for DDA Agent

Stores transitions for PPO training.
"""

import torch
import numpy as np
from typing import Tuple


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO.
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space (for compatibility, not used for DDA)
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.device = device
        
        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.pos = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            value: float, log_prob: float, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        if self.pos >= len(self.states):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
        else:
            self.states[self.pos] = state
            self.actions[self.pos] = action
            self.rewards[self.pos] = reward
            self.values[self.pos] = value
            self.log_probs[self.pos] = log_prob
            self.dones[self.pos] = done
        
        self.pos += 1
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Convert to numpy arrays
        rewards = np.array(self.rewards[:self.pos])
        values = np.array(self.values[:self.pos])
        dones = np.array(self.dones[:self.pos])
        
        # Compute returns and advantages
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        last_gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[step + 1]
            
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            last_gae = delta + gamma * gae_lambda * (1 - dones[step]) * last_gae
            advantages[step] = last_gae
            returns[step] = advantages[step] + values[step]
        
        # Store returns and advantages
        self.returns = returns
        self.advantages = advantages
    
    def get(self, batch_size: int):
        """
        Get mini-batches from the buffer.
        
        Args:
            batch_size: Size of each mini-batch
        
        Yields:
            Batch of (states, actions, old_log_probs, returns, advantages, old_values)
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states[:self.pos])).to(self.device)
        actions_tensor = torch.LongTensor(self.actions[:self.pos]).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs[:self.pos]).to(self.device)
        returns_tensor = torch.FloatTensor(self.returns).to(self.device)
        advantages_tensor = torch.FloatTensor(self.advantages).to(self.device)
        old_values_tensor = torch.FloatTensor(self.values[:self.pos]).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Generate mini-batches
        indices = np.arange(len(self.states[:self.pos]))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                states_tensor[batch_indices],
                actions_tensor[batch_indices],
                old_log_probs_tensor[batch_indices],
                returns_tensor[batch_indices],
                advantages_tensor[batch_indices],
                old_values_tensor[batch_indices]
            )
    
    def reset(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.pos = 0

