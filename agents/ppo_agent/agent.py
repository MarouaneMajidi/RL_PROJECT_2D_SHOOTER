"""
PPO Agent Implementation

Implements the Proximal Policy Optimization algorithm from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional
import os

from .model import ActorCritic
from .memory import RolloutBuffer
from .config import PPOConfig


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    
    Implements PPO with:
    - Clipped surrogate objective
    - Actor-Critic architecture
    - Generalized Advantage Estimation (GAE)
    - Mini-batch updates
    - Gradient clipping
    - Entropy bonus
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize PPO agent.
        
        Args:
            config: PPOConfig object with hyperparameters
        """
        self.config = config
        
        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize actor-critic network with multi-discrete action space
        self.policy = ActorCritic(
            state_dim=config.state_dim,
            action_space_shape=config.action_space_shape,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            device=self.device
        )
        
        # Training statistics
        self.num_updates = 0
        self.total_timesteps = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select actions given the current state (multi-discrete).
        
        Args:
            state: Current state observation
            deterministic: If True, select most likely actions (no sampling)
        
        Returns:
            action: Combined action (movement * 2 + shoot) for compatibility
            log_prob: Log probability of the combined action
            value: State value estimate
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
            
            return action.cpu().item(), log_prob.cpu().item(), value.squeeze().cpu().item()
    
    def decode_action(self, combined_action: int) -> Tuple[int, int]:
        """
        Decode combined action back to (movement, shoot) tuple.
        
        Args:
            combined_action: Combined action value
        
        Returns:
            (movement_action, shoot_action) tuple
        """
        movement = combined_action // self.config.action_space_shape[1]
        shoot = combined_action % self.config.action_space_shape[1]
        return movement, shoot
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """
        Store a transition in the rollout buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        self.rollout_buffer.add(state, action, reward, value, log_prob, done)
        self.total_timesteps += 1
    
    def update(self, last_value: float) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        
        Args:
            last_value: Value estimate for the last state (for bootstrapping)
        
        Returns:
            Dictionary with training statistics
        """
        # Compute returns and advantages using GAE
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        clip_fractions = []
        approx_kl_divs = []
        
        # Perform multiple epochs of updates
        for epoch in range(self.config.n_epochs):
            # Generate mini-batches
            for batch in self.rollout_buffer.get(self.config.batch_size):
                states, actions, old_log_probs, returns, advantages, old_values = batch
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
                values = values.squeeze(-1)
                
                # Compute policy loss with clipped surrogate objective
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss (MSE)
                value_loss = 0.5 * ((returns - values) ** 2).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # Record statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
                
                # Compute clip fraction (fraction of ratios that were clipped)
                with torch.no_grad():
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.config.clip_epsilon).float()).item()
                    clip_fractions.append(clip_fraction)
                    
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                    approx_kl_divs.append(approx_kl)
        
        # Reset buffer
        self.rollout_buffer.reset()
        self.num_updates += 1
        
        # Return training statistics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses),
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': np.mean(approx_kl_divs),
            'num_updates': self.num_updates
        }
    
    def save(self, path: str):
        """
        Save the agent's model and optimizer state.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_updates': self.num_updates,
            'total_timesteps': self.total_timesteps,
            'config': self.config
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the agent's model and optimizer state.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Handle backward compatibility: map old module names to new ones
        import sys
        
        # Map old module paths to new ones for backward compatibility
        old_to_new_modules = {
            'dda_agent': 'agents.dda_agent',
            'ppo_agent': 'agents.ppo_agent',
        }
        
        # Temporarily add old module paths to sys.modules for pickle loading
        temp_modules = {}
        for old_name, new_name in old_to_new_modules.items():
            if old_name not in sys.modules:
                try:
                    # Import the new module and alias it as the old name
                    new_module = __import__(new_name, fromlist=[''])
                    sys.modules[old_name] = new_module
                    temp_modules[old_name] = True
                except ImportError:
                    pass
        
        try:
            # Load with weights_only=False for backwards compatibility with PyTorch 2.6+
            # This is safe as long as you trust the checkpoint source
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        finally:
            # Clean up temporary module aliases
            for old_name in temp_modules:
                if old_name in sys.modules and sys.modules[old_name] is sys.modules.get(old_to_new_modules[old_name]):
                    del sys.modules[old_name]
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_updates = checkpoint.get('num_updates', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        
        print(f"Model loaded from {path}")
        print(f"Timesteps: {self.total_timesteps}, Updates: {self.num_updates}")
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a state.
        
        Args:
            state: State observation
        
        Returns:
            Value estimate
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            value = self.policy.get_value(state_tensor)
            return value.squeeze().cpu().item()
