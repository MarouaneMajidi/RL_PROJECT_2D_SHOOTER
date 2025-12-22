"""
Training Metrics Tracker for PPO Agent

Tracks comprehensive metrics during training:
- Episode rewards (mean, min, max, std)
- Episode lengths
- Win rate / Success rate
- Policy loss, Value loss, Total loss
- Clip fraction, KL divergence, Entropy
- Explained variance
- Learning rate
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque


class TrainingMetricsTracker:
    """
    Comprehensive metrics tracker for PPO training.
    
    Tracks and saves metrics after each episode and at training completion.
    """
    
    def __init__(self, save_dir: str = "checkpoints/player_ppo/metrics"):
        """
        Initialize the metrics tracker.
        
        Args:
            save_dir: Directory to save metrics JSON files
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(save_dir, f"training_metrics_{timestamp}.json")
        
        # Episode-level metrics (reset each episode)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_info = {}
        
        # Accumulated episode metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_scores: List[int] = []  # Game score (increases with kills)
        
        # Training update metrics (updated each PPO update)
        self.update_metrics_history: List[Dict] = []
        
        # Current update metrics (to be saved with episode)
        self.current_update_metrics: Optional[Dict] = None
        
        # Learning rate tracking
        self.learning_rates: List[float] = []
        
        # Initialize metrics storage structure
        self.all_metrics: List[Dict] = []
    
    def start_episode(self):
        """Reset tracking for a new episode."""
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_info = {}
    
    def update_episode_step(self, reward: float, info: Dict):
        """Update episode with a new step."""
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.current_episode_info.update(info)
    
    def end_episode(self, timestep: int, update_metrics: Optional[Dict] = None, 
                    learning_rate: Optional[float] = None):
        """
        Record episode completion and save metrics.
        
        Args:
            timestep: Current total timestep
            update_metrics: Training update metrics (from agent.update())
            learning_rate: Current learning rate
        """
        # Store episode metrics
        # Note: This is an infinite wave survival game, so episodes always end in death
        # Episode length (survival time) and game score are key learning metrics
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        episode_score = self.current_episode_info.get('score', 0)
        self.episode_scores.append(episode_score)
        
        # Store update metrics if available
        if update_metrics:
            self.current_update_metrics = update_metrics.copy()
        
        # Store learning rate
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # Calculate statistics for completed episodes
        episode_count = len(self.episode_rewards)
        
        # Episode reward statistics
        reward_mean = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        reward_min = np.min(self.episode_rewards) if self.episode_rewards else 0.0
        reward_max = np.max(self.episode_rewards) if self.episode_rewards else 0.0
        reward_std = np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0.0
        
        # Episode length statistics (survival time - key learning metric)
        length_mean = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        length_min = np.min(self.episode_lengths) if self.episode_lengths else 0
        length_max = np.max(self.episode_lengths) if self.episode_lengths else 0
        length_std = np.std(self.episode_lengths) if len(self.episode_lengths) > 1 else 0.0
        
        # Calculate average episode length over rolling window (to track learning progress)
        window_size = min(100, len(self.episode_lengths))
        recent_lengths = self.episode_lengths[-window_size:] if self.episode_lengths else []
        avg_survival_time = np.mean(recent_lengths) if recent_lengths else 0.0
        
        # Game score statistics (key learning metric - shows agent is killing zombies, not just surviving)
        score_mean = np.mean(self.episode_scores) if self.episode_scores else 0.0
        score_min = np.min(self.episode_scores) if self.episode_scores else 0
        score_max = np.max(self.episode_scores) if self.episode_scores else 0
        score_std = np.std(self.episode_scores) if len(self.episode_scores) > 1 else 0.0
        
        # Calculate average score over rolling window
        recent_scores = self.episode_scores[-window_size:] if self.episode_scores else []
        avg_score = np.mean(recent_scores) if recent_scores else 0.0
        
        # Prepare metrics dictionary
        episode_metrics = {
            'timestep': timestep,
            'episode': episode_count,
            
            # Episode reward statistics
            'episode_reward': self.current_episode_reward,
            'episode_reward_mean': float(reward_mean),
            'episode_reward_min': float(reward_min),
            'episode_reward_max': float(reward_max),
            'episode_reward_std': float(reward_std),
            
            # Episode length statistics (survival time - key learning metric)
            'episode_length': self.current_episode_length,
            'episode_length_mean': float(length_mean),
            'episode_length_min': int(length_min),
            'episode_length_max': int(length_max),
            'episode_length_std': float(length_std),
            'avg_survival_time': float(avg_survival_time),  # Rolling average survival time
            
            # Game score statistics (key learning metric - shows combat effectiveness)
            'episode_score': int(episode_score),
            'episode_score_mean': float(score_mean),
            'episode_score_min': int(score_min),
            'episode_score_max': int(score_max),
            'episode_score_std': float(score_std),
            'avg_score': float(avg_score),  # Rolling average score
            
            # Training update metrics (from last PPO update)
            'policy_loss': float(self.current_update_metrics.get('policy_loss', 0.0)) if self.current_update_metrics else 0.0,
            'value_loss': float(self.current_update_metrics.get('value_loss', 0.0)) if self.current_update_metrics else 0.0,
            'total_loss': float(self.current_update_metrics.get('total_loss', 0.0)) if self.current_update_metrics else 0.0,
            'clip_fraction': float(self.current_update_metrics.get('clip_fraction', 0.0)) if self.current_update_metrics else 0.0,
            'kl_divergence': float(self.current_update_metrics.get('approx_kl', 0.0)) if self.current_update_metrics else 0.0,
            'entropy': float(-self.current_update_metrics.get('entropy_loss', 0.0)) if self.current_update_metrics else 0.0,  # Convert loss to entropy
            'explained_variance': float(self.current_update_metrics.get('explained_variance', 0.0)) if self.current_update_metrics else 0.0,
            
            # Learning rate
            'learning_rate': float(learning_rate) if learning_rate is not None else 0.0,
            
            # Additional episode info
            'zombies_killed': self.current_episode_info.get('zombies_killed', 0),
            'wave': self.current_episode_info.get('wave', 1),
        }
        
        # Add to history
        self.all_metrics.append(episode_metrics)
        
        # Save to JSON file after each episode
        self._save_metrics()
    
    def record_update(self, update_metrics: Dict, learning_rate: float):
        """
        Record training update metrics.
        
        Args:
            update_metrics: Dictionary with training statistics from agent.update()
            learning_rate: Current learning rate
        """
        self.current_update_metrics = update_metrics.copy()
        self.learning_rates.append(learning_rate)
    
    def finalize(self, total_timesteps: int):
        """
        Finalize metrics tracking and save final metrics.
        
        Args:
            total_timesteps: Total timesteps trained
        """
        # Create summary statistics
        summary = {
            'total_timesteps': total_timesteps,
            'total_episodes': len(self.episode_rewards),
            'final_reward_mean': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'final_reward_std': float(np.std(self.episode_rewards)) if len(self.episode_rewards) > 1 else 0.0,
            'final_episode_length_mean': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            'final_episode_length_std': float(np.std(self.episode_lengths)) if len(self.episode_lengths) > 1 else 0.0,
            'final_episode_length_max': int(np.max(self.episode_lengths)) if self.episode_lengths else 0,
            'final_score_mean': float(np.mean(self.episode_scores)) if self.episode_scores else 0.0,
            'final_score_std': float(np.std(self.episode_scores)) if len(self.episode_scores) > 1 else 0.0,
            'final_score_max': int(np.max(self.episode_scores)) if self.episode_scores else 0,
        }
        
        # Add summary to metrics
        final_metrics = {
            'summary': summary,
            'episodes': self.all_metrics
        }
        
        # Save final metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {self.metrics_file}")
        print(f"Total episodes tracked: {len(self.episode_rewards)}")
        print(f"Final survival time (episode length): {summary['final_episode_length_mean']:.1f} ± {summary['final_episode_length_std']:.1f} steps")
        print(f"Best survival time: {summary['final_episode_length_max']} steps")
        print(f"Final game score: {summary['final_score_mean']:.1f} ± {summary['final_score_std']:.1f} points")
        print(f"Best game score: {summary['final_score_max']} points")
        print(f"Final reward mean: {summary['final_reward_mean']:.2f} ± {summary['final_reward_std']:.2f}")
    
    def _save_metrics(self):
        """Save current metrics to JSON file."""
        metrics_dict = {
            'episodes': self.all_metrics
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def get_latest_metrics(self) -> Dict:
        """Get the most recent episode metrics."""
        if self.all_metrics:
            return self.all_metrics[-1]
        return {}