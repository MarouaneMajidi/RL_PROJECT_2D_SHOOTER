"""
CNN-based PPO Agent for DDA

Uses CNN to process raw RGB images and PPO algorithm for training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional
import os

from .model import CNNDDAActorCritic
from .memory import DDACNNRolloutBuffer
from .config import CNNDDAConfig


class CNNDDAAgent:
    """
    CNN-based PPO Agent for Dynamic Difficulty Adjustment.
    
    Uses convolutional neural network to extract features from raw RGB images.
    Implements PPO algorithm for training.
    """
    
    def __init__(self, config: CNNDDAConfig):
        """
        Initialize CNN-based DDA agent.
        
        Args:
            config: CNNDDAConfig object with hyperparameters
        """
        self.config = config
        
        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"CNN DDA Agent using device: {self.device}")
        
        # Initialize CNN actor-critic network
        self.policy = CNNDDAActorCritic(
            input_channels=config.CNN_INPUT_CHANNELS,  # 12
            action_dim=config.action_dim,
            feature_dim=config.feature_dim,
            cnn_hidden_dim=config.cnn_hidden_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Initialize rollout buffer with image state shape
        state_shape = (config.CNN_INPUT_CHANNELS, config.RESIZE_HEIGHT, config.RESIZE_WIDTH)  # (12, 96, 128)
        self.rollout_buffer = DDACNNRolloutBuffer(
            buffer_size=config.n_steps,
            state_shape=state_shape,
            action_dim=config.action_dim,
            device=self.device
        )
        
        # Training statistics
        self.num_updates = 0
        self.total_timesteps = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action given the current state image.
        
        Args:
            state: Current state image (12, 96, 128) as numpy array
            deterministic: If True, select most likely action (no sampling)
        
        Returns:
            action: Selected action (0-4)
            log_prob: Log probability of the action
            value: State value estimate
        """
        with torch.no_grad():
            # Convert numpy to tensor and add batch dimension if needed
            state_tensor = torch.from_numpy(state).float().to(self.device)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
            
            return action.cpu().item(), log_prob.cpu().item(), value.squeeze().cpu().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """
        Store a transition in the rollout buffer.
        
        Args:
            state: Current state image (12, 96, 128)
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
        
        # For explained variance calculation
        all_returns = []
        all_values = []
        
        # Perform multiple epochs of updates
        for epoch in range(self.config.n_epochs):
            # Generate mini-batches
            for batch in self.rollout_buffer.get(self.config.batch_size):
                states, actions, old_log_probs, returns, advantages, old_values = batch
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
                values = values.squeeze(-1)
                
                # Store returns and values for explained variance (only in first epoch)
                if epoch == 0:
                    all_returns.append(returns.detach().cpu().numpy())
                    all_values.append(values.detach().cpu().numpy())
                
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
                
                # Compute clip fraction
                with torch.no_grad():
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.config.clip_epsilon).float()).item()
                    clip_fractions.append(clip_fraction)
                    
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                    approx_kl_divs.append(approx_kl)
        
        # Calculate explained variance
        explained_variance = 0.0
        if all_returns and all_values:
            returns_array = np.concatenate(all_returns)
            values_array = np.concatenate(all_values)
            returns_var = np.var(returns_array)
            if returns_var > 1e-8:
                explained_variance = 1.0 - np.var(returns_array - values_array) / returns_var
            else:
                explained_variance = 0.0
        
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
            'explained_variance': explained_variance,
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
        
        print(f"CNN DDA Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the agent's model and optimizer state.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_updates = checkpoint.get('num_updates', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        
        print(f"CNN DDA Model loaded from {path}")
        print(f"Timesteps: {self.total_timesteps}, Updates: {self.num_updates}")
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a state image.
        
        Args:
            state: State image (12, 96, 128)
        
        Returns:
            Value estimate
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            value = self.policy.get_value(state_tensor)
            return value.squeeze().cpu().item()
