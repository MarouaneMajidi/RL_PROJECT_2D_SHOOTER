"""
CNN-based Actor-Critic Model for DDA Agent

Uses convolutional neural network to extract features from raw RGB images.
Input: 4 stacked frames (128×96 RGB) = 12 channels (4 frames × 3 RGB)
Output: Actor and Critic heads for PPO with discrete action space (5 actions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNDDAActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for DDA PPO.
    
    Architecture:
    - Input: (batch, 12, 96, 128) where 12 = 4 frames × 3 RGB channels
    - CNN feature extraction (conv layers)
    - Flatten → Dense layers
    - Actor head: outputs action logits for discrete action space (5 actions)
    - Critic head: outputs value estimate
    """
    
    def __init__(self, 
                 input_channels: int = 12,  # 4 frames × 3 RGB channels
                 action_dim: int = 5,  # DDA actions: 0=do nothing, 1-2=easier, 3-4=harder
                 feature_dim: int = 128,
                 cnn_hidden_dim: int = 64):
        """
        Initialize CNN Actor-Critic network for DDA.
        
        Args:
            input_channels: Number of input channels (4 frames × 3 RGB = 12)
            action_dim: Number of discrete actions (5 for DDA)
            feature_dim: Dimension of features after CNN extraction
            cnn_hidden_dim: Hidden dimension in CNN layers
        """
        super(CNNDDAActorCritic, self).__init__()
        
        self.action_dim = action_dim
        self.input_channels = input_channels
        
        # CNN Feature Extraction (same architecture as player CNN)
        # Input: (batch, 12, 96, 128)
        self.conv1 = nn.Conv2d(input_channels, cnn_hidden_dim, kernel_size=8, stride=4, padding=2)
        # Output: (batch, 64, 24, 32)
        
        self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        # Output: (batch, 128, 12, 16)
        
        self.conv3 = nn.Conv2d(cnn_hidden_dim * 2, cnn_hidden_dim * 2, kernel_size=3, stride=1, padding=1)
        # Output: (batch, 128, 12, 16)
        
        # Calculate flattened size: 128 * 12 * 16 = 24576
        self.flattened_size = cnn_hidden_dim * 2 * 12 * 16
        
        # Dense layers (smaller for DDA)
        self.fc1 = nn.Linear(self.flattened_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        
        # Actor head (discrete action space: 5 actions)
        self.actor = nn.Linear(feature_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(feature_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128)
               representing 4 stacked RGB frames
        
        Returns:
            action_logits: Action logits for discrete actions (batch, action_dim)
            values: Value estimates (batch, 1)
        """
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor head (discrete actions)
        action_logits = self.actor(x)
        
        # Critic head
        values = self.critic(x)
        
        return action_logits, values
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128) or (12, 96, 128)
            deterministic: If True, select most likely action (no sampling)
        
        Returns:
            action: Action index (0-4)
            log_prob: Log probability of the action
            value: Value estimate
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Ensure correct shape: (batch, channels, height, width)
        if x.shape[-3:] != (12, 96, 128):
            raise ValueError(f"Expected input shape (..., 12, 96, 128), got {x.shape}")
        
        action_logits, values = self.forward(x)
        
        # Sample action from categorical distribution
        from torch.distributions import Categorical
        
        dist = Categorical(logits=action_logits)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, values.squeeze(-1)
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states (used during training).
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128)
            actions: Action tensor of shape (batch,) with action indices (0-4)
        
        Returns:
            log_probs: Log probabilities of actions (batch,)
            values: State values (batch, 1)
            entropy: Entropy of action distribution (batch,)
        """
        action_logits, values = self.forward(x)
        
        from torch.distributions import Categorical
        
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for a state.
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128) or (12, 96, 128)
        
        Returns:
            Value estimate (batch, 1) or scalar
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        _, values = self.forward(x)
        return values.squeeze(-1)
