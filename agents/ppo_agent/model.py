"""
Actor-Critic Neural Network Model - REDESIGNED FOR MULTI-DISCRETE ACTIONS

Supports multi-discrete action space:
- Movement action: 5 options (up, down, left, right, idle)
- Shoot action: 2 options (shoot, don't shoot)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with multi-discrete action space.
    
    The network has:
    - Shared feature extractor
    - Separate actor heads for movement and shoot actions
    - Critic head (value function)
    """
    
    def __init__(self, state_dim: int, action_space_shape: Tuple[int, ...], hidden_dim: int = 256):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_space_shape: Tuple of action space dimensions, e.g., (5, 2) for [movement, shoot]
            hidden_dim: Dimension of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_space_shape = action_space_shape
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate actor heads for each action dimension
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
            for action_dim in action_space_shape
        ])
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Last layers with smaller gain
        for actor_head in self.actor_heads:
            nn.init.orthogonal_(actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_logits_list: List of logits for each action dimension
            value: State value estimate (batch_size, 1)
        """
        # Extract shared features
        features = self.shared_net(state)
        
        # Get action logits for each dimension
        action_logits_list = [actor_head(features) for actor_head in self.actor_heads]
        
        # Get value
        value = self.critic(features)
        
        return action_logits_list, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy for multi-discrete action space.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
            deterministic: If True, return the most likely actions (no sampling)
        
        Returns:
            actions: Sampled actions as tuple of tensors, one per action dimension
            log_prob: Sum of log probabilities of all actions
            value: State value estimate
        """
        # Handle single state input
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Forward pass
        action_logits_list, value = self.forward(state)
        
        # Sample actions for each dimension
        actions = []
        log_probs = []
        
        for action_logits in action_logits_list:
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            actions.append(action)
        
        # Combine actions into a single tensor for storage (flattened index)
        # For (5, 2) action space: action = movement * 2 + shoot
        combined_action = actions[0] * self.action_space_shape[1] + actions[1]
        
        # Sum log probabilities (independent actions)
        total_log_prob = sum(log_probs)
        
        # Squeeze if single state was provided
        if squeeze_output:
            combined_action = combined_action.squeeze(0)
            total_log_prob = total_log_prob.squeeze(0)
        
        return combined_action, total_log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states (used during training).
        
        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Combined action tensor of shape (batch_size,)
                    Actions are encoded as: movement * action_space_shape[1] + shoot
        
        Returns:
            log_probs: Log probabilities of actions (batch_size,)
            values: State values (batch_size, 1)
            entropy: Sum of entropies of action distributions (batch_size,)
        """
        # Forward pass
        action_logits_list, values = self.forward(states)
        
        # Decode combined actions back to separate actions
        movement_actions = actions // self.action_space_shape[1]
        shoot_actions = actions % self.action_space_shape[1]
        decoded_actions = [movement_actions, shoot_actions]
        
        # Evaluate log probabilities and entropy for each action dimension
        log_probs = []
        entropies = []
        
        for i, action_logits in enumerate(action_logits_list):
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            log_probs.append(dist.log_prob(decoded_actions[i]))
            entropies.append(dist.entropy())
        
        # Sum log probabilities and entropies
        total_log_prob = sum(log_probs)
        total_entropy = sum(entropies)
        
        return total_log_prob, values, total_entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for a state.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            value: State value estimate
        """
        # Handle single state input
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get features and value
        features = self.shared_net(state)
        value = self.critic(features)
        
        return value
