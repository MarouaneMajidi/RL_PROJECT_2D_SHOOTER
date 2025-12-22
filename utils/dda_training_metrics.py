"""
Training Metrics Tracker for DDA Agent

Tracks comprehensive metrics during DDA training:
- Episode rewards (mean, min, max, std)
- Episode lengths
- Policy loss, Value loss, Total loss
- Clip fraction, KL divergence, Entropy
- Explained variance
- Learning rate
- Difficulty levels being set (mean, min, max per episode)
- Player performance/score under DDA
- Difficulty adjustment frequency
- Player engagement/challenge metrics
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque


class DDATrainingMetricsTracker:
    """
    Comprehensive metrics tracker for DDA training.
    
    Tracks and saves metrics after each episode and at training completion.
    """
    
    def __init__(self, save_dir: str = "checkpoints/dda/metrics"):
        """
        Initialize the metrics tracker.
        
        Args:
            save_dir: Directory to save metrics JSON files
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(save_dir, f"dda_training_metrics_{timestamp}.json")
        
        # Episode-level metrics (reset each episode)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_info = {}
        
        # Track difficulty adjustments during episode
        self.episode_difficulty_params_history: List[Dict] = []
        self.episode_dda_actions: List[int] = []
        self.episode_adjustment_count = 0  # Count of non-zero actions (actual adjustments)
        
        # Accumulated episode metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Player performance metrics per episode
        self.episode_player_scores: List[int] = []  # Game score or kills
        self.episode_player_healths: List[float] = []  # Final health
        self.episode_player_kills: List[int] = []  # Zombies killed
        
        # Difficulty metrics per episode
        self.episode_difficulty_levels: List[float] = []  # Aggregate difficulty score
        self.episode_adjustment_frequencies: List[float] = []  # Adjustments per episode length
        
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
        self.episode_difficulty_params_history = []
        self.episode_dda_actions = []
        self.episode_adjustment_count = 0
    
    def update_episode_step(self, reward: float, info: Dict, dda_action: Optional[int] = None):
        """
        Update episode with a new step.
        
        Args:
            reward: Reward from DDA environment
            info: Info dict from DDA environment step
            dda_action: DDA action taken (optional, for tracking adjustment frequency)
        """
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.current_episode_info.update(info)
        
        # Track difficulty parameters if available
        if 'difficulty_params' in info:
            self.episode_difficulty_params_history.append(info['difficulty_params'].copy())
        
        # Track DDA actions for adjustment frequency
        if dda_action is not None:
            self.episode_dda_actions.append(dda_action)
            if dda_action != 0:  # Non-zero means an adjustment was made
                self.episode_adjustment_count += 1
    
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
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        # Extract player performance metrics
        player_score = self.current_episode_info.get('zombies_killed', 0)  # Use kills as score proxy
        player_health = self.current_episode_info.get('player_health', 0.0)
        player_kills = self.current_episode_info.get('zombies_killed', 0)
        
        self.episode_player_scores.append(player_score)
        self.episode_player_healths.append(player_health)
        self.episode_player_kills.append(player_kills)
        
        # Calculate difficulty metrics
        difficulty_level = self._calculate_difficulty_level()
        self.episode_difficulty_levels.append(difficulty_level)
        
        # Calculate adjustment frequency (adjustments per episode length)
        adjustment_freq = self.episode_adjustment_count / max(self.current_episode_length, 1)
        self.episode_adjustment_frequencies.append(adjustment_freq)
        
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
        
        # Episode length statistics
        length_mean = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        length_min = np.min(self.episode_lengths) if self.episode_lengths else 0
        length_max = np.max(self.episode_lengths) if self.episode_lengths else 0
        length_std = np.std(self.episode_lengths) if len(self.episode_lengths) > 1 else 0.0
        
        # Player performance statistics
        score_mean = np.mean(self.episode_player_scores) if self.episode_player_scores else 0.0
        score_max = np.max(self.episode_player_scores) if self.episode_player_scores else 0
        
        # Difficulty level statistics
        diff_mean = np.mean(self.episode_difficulty_levels) if self.episode_difficulty_levels else 0.0
        diff_min = np.min(self.episode_difficulty_levels) if self.episode_difficulty_levels else 0.0
        diff_max = np.max(self.episode_difficulty_levels) if self.episode_difficulty_levels else 0.0
        
        # Adjustment frequency statistics
        adj_freq_mean = np.mean(self.episode_adjustment_frequencies) if self.episode_adjustment_frequencies else 0.0
        
        # Get difficulty params stats for this episode
        diff_params_stats = self._calculate_difficulty_params_stats()
        
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
            
            # Episode length statistics
            'episode_length': self.current_episode_length,
            'episode_length_mean': float(length_mean),
            'episode_length_min': int(length_min),
            'episode_length_max': int(length_max),
            'episode_length_std': float(length_std),
            
            # Training update metrics (from last PPO update)
            'policy_loss': float(self.current_update_metrics.get('policy_loss', 0.0)) if self.current_update_metrics else 0.0,
            'value_loss': float(self.current_update_metrics.get('value_loss', 0.0)) if self.current_update_metrics else 0.0,
            'total_loss': float(self.current_update_metrics.get('total_loss', 0.0)) if self.current_update_metrics else 0.0,
            'clip_fraction': float(self.current_update_metrics.get('clip_fraction', 0.0)) if self.current_update_metrics else 0.0,
            'kl_divergence': float(self.current_update_metrics.get('approx_kl', 0.0)) if self.current_update_metrics else 0.0,
            'entropy': float(-self.current_update_metrics.get('entropy_loss', 0.0)) if self.current_update_metrics else 0.0,
            'explained_variance': float(self.current_update_metrics.get('explained_variance', 0.0)) if self.current_update_metrics else 0.0,
            
            # Learning rate
            'learning_rate': float(learning_rate) if learning_rate is not None else 0.0,
            
            # Player performance under DDA
            'player_score': int(player_score),
            'player_score_mean': float(score_mean),
            'player_score_max': int(score_max),
            'player_health': float(player_health),
            'player_kills': int(player_kills),
            
            # Difficulty level metrics
            'difficulty_level': float(difficulty_level),
            'difficulty_level_mean': float(diff_mean),
            'difficulty_level_min': float(diff_min),
            'difficulty_level_max': float(diff_max),
            
            # Difficulty adjustment frequency
            'adjustment_count': int(self.episode_adjustment_count),
            'adjustment_frequency': float(adjustment_freq),
            'adjustment_frequency_mean': float(adj_freq_mean),
            
            # Difficulty parameters (mean values for this episode)
            'difficulty_params': diff_params_stats,
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
    
    def _calculate_difficulty_level(self) -> float:
        """
        Calculate an aggregate difficulty level from difficulty parameters.
        
        Returns:
            Aggregate difficulty score (0.0 to 1.0, where 1.0 is hardest)
        """
        if not self.episode_difficulty_params_history:
            return 0.5  # Default to medium
        
        # Get average difficulty params over episode
        avg_params = {}
        for key in self.episode_difficulty_params_history[0].keys():
            values = [params.get(key, 0) for params in self.episode_difficulty_params_history]
            avg_params[key] = np.mean(values) if values else 0
        
        # Normalize parameters to [0, 1] based on expected ranges
        # Higher values = harder (for enemy params), lower = harder (for player/helpful params)
        difficulty_components = []
        
        # Enemy pressure (higher = harder)
        if 'ZOMBIE_SPEED' in avg_params:
            difficulty_components.append(np.clip((avg_params['ZOMBIE_SPEED'] - 1.0) / 4.0, 0, 1))
        if 'SPAWN_RATE' in avg_params:
            # Lower spawn rate = faster spawn = harder (inverted)
            difficulty_components.append(np.clip(1.0 - (avg_params['SPAWN_RATE'] - 10) / 110.0, 0, 1))
        
        # Enemy durability (higher = harder)
        if 'ZOMBIE_NORMAL_HEALTH' in avg_params:
            difficulty_components.append(np.clip((avg_params['ZOMBIE_NORMAL_HEALTH'] - 20) / 30.0, 0, 1))
        
        # Player power (lower = harder, so invert)
        if 'PLAYER_SPEED' in avg_params:
            difficulty_components.append(np.clip(1.0 - (avg_params['PLAYER_SPEED'] - 3.0) / 5.0, 0, 1))
        if 'PISTOL_DAMAGE' in avg_params:
            difficulty_components.append(np.clip(1.0 - (avg_params['PISTOL_DAMAGE'] - 15) / 15.0, 0, 1))
        
        # Return average difficulty
        return np.mean(difficulty_components) if difficulty_components else 0.5
    
    def _calculate_difficulty_params_stats(self) -> Dict:
        """
        Calculate statistics (mean, min, max) for difficulty parameters over the episode.
        
        Returns:
            Dictionary with parameter statistics
        """
        if not self.episode_difficulty_params_history:
            return {}
        
        stats = {}
        
        # Get all parameter keys
        param_keys = set()
        for params in self.episode_difficulty_params_history:
            param_keys.update(params.keys())
        
        # Calculate stats for each parameter
        for key in param_keys:
            values = [params.get(key, 0) for params in self.episode_difficulty_params_history]
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_min'] = float(np.min(values))
                stats[f'{key}_max'] = float(np.max(values))
        
        return stats
    
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
            'final_player_score_mean': float(np.mean(self.episode_player_scores)) if self.episode_player_scores else 0.0,
            'final_difficulty_level_mean': float(np.mean(self.episode_difficulty_levels)) if self.episode_difficulty_levels else 0.0,
            'final_adjustment_frequency_mean': float(np.mean(self.episode_adjustment_frequencies)) if self.episode_adjustment_frequencies else 0.0,
        }
        
        # Add summary to metrics
        final_metrics = {
            'summary': summary,
            'episodes': self.all_metrics
        }
        
        # Save final metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\nDDA Metrics saved to: {self.metrics_file}")
        print(f"Total episodes tracked: {len(self.episode_rewards)}")
        print(f"Final reward mean: {summary['final_reward_mean']:.2f} ± {summary['final_reward_std']:.2f}")
        print(f"Final player score mean: {summary['final_player_score_mean']:.1f}")
        print(f"Final difficulty level mean: {summary['final_difficulty_level_mean']:.3f}")
        print(f"Final adjustment frequency mean: {summary['final_adjustment_frequency_mean']:.4f}")
    
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
