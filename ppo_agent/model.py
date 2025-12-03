"""
Actor-Critic Neural Network Model

Implements the neural network architecture for PPO with separate
actor (policy) and critic (value) heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    The network has:
    - Shared feature extractor
    - Actor head (policy) that outputs action probabilities
    - Critic head (value function) that outputs state value
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space (number of discrete actions)
            hidden_dim: Dimension of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
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
        
        # Last layer of actor and critic with smaller gain
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_logits: Logits for action distribution (batch_size, action_dim)
            value: State value estimate (batch_size, 1)
        """
        # Extract shared features
        features = self.shared_net(state)
        
        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
            deterministic: If True, return the most likely action (no sampling)
        
        Returns:
            action: Sampled action (batch_size,) or scalar
            log_prob: Log probability of the action (batch_size,) or scalar
            value: State value estimate (batch_size, 1) or (1,)
        """
        # Handle single state input
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Forward pass
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Sample or take most likely action
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        # Squeeze if single state was provided
        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states (used during training).
        
        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size,)
        
        Returns:
            log_probs: Log probabilities of actions (batch_size,)
            values: State values (batch_size, 1)
            entropy: Entropy of the action distribution (batch_size,)
        """
        # Forward pass
        action_logits, values = self.forward(states)
        
        # Create categorical distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Evaluate log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy
    
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
