"""
CNN-based Actor-Critic Model for Player Agent

Uses convolutional neural network to extract features from raw RGB images.
Input: 4 stacked frames (128×96 RGB) = 12 channels (4 frames × 3 RGB)
Output: Actor and Critic heads for PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for PPO.
    
    Architecture:
    - Input: (batch, 12, 96, 128) where 12 = 4 frames × 3 RGB channels
    - CNN feature extraction (conv layers)
    - Flatten → Dense layers
    - Actor head: outputs action logits for multi-discrete action space
    - Critic head: outputs value estimate
    """
    
    def __init__(self, 
                 input_channels: int = 12,  # 4 frames × 3 RGB channels
                 action_space_shape: Tuple[int, ...] = (5, 2),
                 feature_dim: int = 256,
                 cnn_hidden_dim: int = 64):
        """
        Initialize CNN Actor-Critic network.
        
        Args:
            input_channels: Number of input channels (4 frames × 3 RGB = 12)
            action_space_shape: Multi-discrete action space shape (movement, shoot)
            feature_dim: Dimension of features after CNN extraction
            cnn_hidden_dim: Hidden dimension in CNN layers
        """
        super(CNNActorCritic, self).__init__()
        
        self.action_space_shape = action_space_shape
        self.input_channels = input_channels
        
        # CNN Feature Extraction
        # Input: (batch, 12, 96, 128)
        self.conv1 = nn.Conv2d(input_channels, cnn_hidden_dim, kernel_size=8, stride=4, padding=2)
        # Output: (batch, 64, 24, 32)
        
        self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        # Output: (batch, 128, 12, 16)
        
        self.conv3 = nn.Conv2d(cnn_hidden_dim * 2, cnn_hidden_dim * 2, kernel_size=3, stride=1, padding=1)
        # Output: (batch, 128, 12, 16)
        
        # Calculate flattened size: 128 * 12 * 16 = 24576
        self.flattened_size = cnn_hidden_dim * 2 * 12 * 16
        
        # Dense layers
        self.fc1 = nn.Linear(self.flattened_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        
        # Actor head (for multi-discrete action space)
        # Movement: 5 actions, Shoot: 2 actions
        self.actor_heads = nn.ModuleList([
            nn.Linear(feature_dim, action_space_shape[0]),  # Movement head
            nn.Linear(feature_dim, action_space_shape[1])   # Shoot head
        ])
        
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
    
    def forward(self, x: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128)
               representing 4 stacked RGB frames
        
        Returns:
            action_logits_list: List of action logits for each action dimension
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
        
        # Actor heads (multi-discrete)
        action_logits_list = [head(x) for head in self.actor_heads]
        
        # Critic head
        values = self.critic(x)
        
        return action_logits_list, values
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128) or (12, 96, 128)
            deterministic: If True, select most likely actions (no sampling)
        
        Returns:
            combined_action: Combined action value (movement * 2 + shoot)
            log_prob: Log probability of the action
            value: Value estimate
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Ensure correct shape: (batch, channels, height, width)
        # Input should be (batch, 12, 96, 128) where 12 = 4 frames × 3 RGB
        if x.shape[-3:] != (12, 96, 128):
            raise ValueError(f"Expected input shape (..., 12, 96, 128), got {x.shape}")
        
        action_logits_list, values = self.forward(x)
        
        # Sample actions from each head
        from torch.distributions import Categorical
        
        actions = []
        log_probs = []
        entropies = []
        
        for logits in action_logits_list:
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        # Combine actions: movement * 2 + shoot
        movement_action = actions[0]
        shoot_action = actions[1]
        combined_action = movement_action * self.action_space_shape[1] + shoot_action
        
        # Sum log probabilities and entropies
        total_log_prob = log_probs[0] + log_probs[1]
        total_entropy = entropies[0] + entropies[1]
        
        return combined_action, total_log_prob, values.squeeze(-1)
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states (used during training).
        
        Args:
            x: Input tensor of shape (batch, 12, 96, 128)
            actions: Combined action tensor of shape (batch,)
                    Actions are encoded as: movement * action_space_shape[1] + shoot
        
        Returns:
            log_probs: Log probabilities of actions (batch,)
            values: State values (batch, 1)
            entropy: Sum of entropies of action distributions (batch,)
        """
        action_logits_list, values = self.forward(x)
        
        # Decode combined actions back to separate actions
        movement_actions = actions // self.action_space_shape[1]
        shoot_actions = actions % self.action_space_shape[1]
        decoded_actions = [movement_actions, shoot_actions]
        
        # Evaluate log probabilities and entropy for each action dimension
        from torch.distributions import Categorical
        
        log_probs = []
        entropies = []
        
        for i, action_logits in enumerate(action_logits_list):
            dist = Categorical(logits=action_logits)
            log_probs.append(dist.log_prob(decoded_actions[i]))
            entropies.append(dist.entropy())
        
        # Sum log probabilities and entropies
        total_log_prob = sum(log_probs)
        total_entropy = sum(entropies)
        
        return total_log_prob, values, total_entropy
    
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
