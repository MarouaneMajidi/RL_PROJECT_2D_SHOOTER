"""
Visualization Script for PPO Training Metrics

Loads metrics from JSON file and creates comprehensive visualizations
of the training process.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_metrics(metrics_file: str) -> dict:
    """
    Load metrics from JSON file.
    
    Args:
        metrics_file: Path to metrics JSON file
    
    Returns:
        Dictionary with metrics data
    """
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return data


def smooth_data(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Smooth data using a moving average.
    
    Args:
        data: Array of data points
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed data array
    """
    if len(data) < window_size:
        return data
    
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    # Pad with first/last values to maintain length
    pad_size = window_size // 2
    smoothed = np.pad(smoothed, (pad_size, window_size - pad_size - 1), mode='edge')
    
    return smoothed


def create_visualizations(metrics_data: dict, output_dir: str = None, smooth_window: int = 100):
    """
    Create comprehensive visualizations of training metrics.
    
    Args:
        metrics_data: Dictionary containing metrics data
        output_dir: Directory to save plots (if None, displays them)
        smooth_window: Window size for smoothing curves
    """
    episodes = metrics_data.get('episodes', [])
    
    if not episodes:
        print("No episode data found in metrics file!")
        return
    
    # Extract data
    timesteps = [ep['timestep'] for ep in episodes]
    episode_nums = [ep['episode'] for ep in episodes]
    
    # Episode rewards
    episode_rewards = [ep['episode_reward'] for ep in episodes]
    reward_means = [ep['episode_reward_mean'] for ep in episodes]
    reward_mins = [ep['episode_reward_min'] for ep in episodes]
    reward_maxs = [ep['episode_reward_max'] for ep in episodes]
    
    # Episode lengths (survival time - key learning metric)
    episode_lengths = [ep['episode_length'] for ep in episodes]
    length_means = [ep['episode_length_mean'] for ep in episodes]
    avg_survival_times = [ep.get('avg_survival_time', ep.get('episode_length_mean', 0)) for ep in episodes]
    
    # Game scores (key learning metric - shows combat effectiveness)
    episode_scores = [ep.get('episode_score', 0) for ep in episodes]
    score_means = [ep.get('episode_score_mean', 0) for ep in episodes]
    score_mins = [ep.get('episode_score_min', 0) for ep in episodes]
    score_maxs = [ep.get('episode_score_max', 0) for ep in episodes]
    avg_scores = [ep.get('avg_score', ep.get('episode_score_mean', 0)) for ep in episodes]
    
    # Losses
    policy_losses = [ep['policy_loss'] for ep in episodes]
    value_losses = [ep['value_loss'] for ep in episodes]
    total_losses = [ep['total_loss'] for ep in episodes]
    
    # Other metrics
    clip_fractions = [ep['clip_fraction'] for ep in episodes]
    kl_divergences = [ep['kl_divergence'] for ep in episodes]
    entropies = [ep['entropy'] for ep in episodes]
    explained_variances = [ep['explained_variance'] for ep in episodes]
    learning_rates = [ep['learning_rate'] for ep in episodes]
    
    # Convert to numpy arrays for smoothing
    episode_rewards = np.array(episode_rewards)
    reward_means = np.array(reward_means)
    episode_lengths = np.array(episode_lengths)
    avg_survival_times = np.array(avg_survival_times)
    episode_scores = np.array(episode_scores)
    score_means = np.array(score_means)
    avg_scores = np.array(avg_scores)
    policy_losses = np.array(policy_losses)
    value_losses = np.array(value_losses)
    total_losses = np.array(total_losses)
    clip_fractions = np.array(clip_fractions)
    kl_divergences = np.array(kl_divergences)
    entropies = np.array(entropies)
    explained_variances = np.array(explained_variances)
    learning_rates = np.array(learning_rates)
    
    # Create figure with subplots (5 rows x 3 cols for more plots)
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
    
    # 1. Episode Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episode_nums, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(episode_nums, smooth_data(episode_rewards, smooth_window), 
             color='blue', linewidth=2, label=f'Smoothed (window={smooth_window})')
    ax1.plot(episode_nums, reward_means, color='red', linewidth=2, label='Mean Reward')
    ax1.fill_between(episode_nums, reward_mins, reward_maxs, alpha=0.2, color='gray', label='Min/Max Range')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Reward Statistics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episode_nums, reward_means, label='Mean', linewidth=2)
    ax2.plot(episode_nums, reward_mins, label='Min', linewidth=1, alpha=0.7)
    ax2.plot(episode_nums, reward_maxs, label='Max', linewidth=1, alpha=0.7)
    reward_stds = [ep['episode_reward_std'] for ep in episodes]
    ax2.fill_between(episode_nums, 
                     np.array(reward_means) - np.array(reward_stds),
                     np.array(reward_means) + np.array(reward_stds),
                     alpha=0.2, label='±1 Std Dev')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Statistics (Mean, Min, Max, Std)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Episode Lengths (Survival Time - Key Learning Metric)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(episode_nums, episode_lengths, alpha=0.3, color='green', label='Episode Length (Survival Time)')
    ax3.plot(episode_nums, smooth_data(episode_lengths, smooth_window), 
             color='green', linewidth=2, label=f'Smoothed (window={smooth_window})')
    ax3.plot(episode_nums, length_means, color='orange', linewidth=2, label='Mean Survival Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Survival Time (steps)')
    ax3.set_title('Episode Length / Survival Time (Key Learning Metric)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Game Score (Key Learning Metric - Combat Effectiveness)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(episode_nums, episode_scores, alpha=0.3, color='red', label='Episode Score')
    ax4.plot(episode_nums, smooth_data(episode_scores, smooth_window), 
             color='red', linewidth=2, label=f'Smoothed (window={smooth_window})')
    ax4.plot(episode_nums, score_means, color='darkred', linewidth=2, label='Mean Score')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Game Score (points)')
    ax4.set_title('Game Score (Key Learning Metric - Combat Effectiveness)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Average Score and Survival Time Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    # Normalize both for comparison (use percentage of max)
    if len(avg_scores) > 0 and np.max(avg_scores) > 0:
        normalized_scores = (avg_scores / np.max(avg_scores)) * 100
    else:
        normalized_scores = avg_scores
    if len(avg_survival_times) > 0 and np.max(avg_survival_times) > 0:
        normalized_survival = (avg_survival_times / np.max(avg_survival_times)) * 100
    else:
        normalized_survival = avg_survival_times
    ax5.plot(episode_nums, normalized_scores, linewidth=2, color='red', label='Avg Score (normalized %)')
    ax5.plot(episode_nums, normalized_survival, linewidth=2, color='green', label='Avg Survival Time (normalized %)')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Normalized Value (%)')
    ax5.set_title('Score vs Survival Time (Normalized Comparison)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Losses
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(episode_nums, smooth_data(policy_losses, smooth_window), 
             label='Policy Loss', linewidth=2)
    ax6.plot(episode_nums, smooth_data(value_losses, smooth_window), 
             label='Value Loss', linewidth=2)
    ax6.plot(episode_nums, smooth_data(total_losses, smooth_window), 
             label='Total Loss', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Loss')
    ax6.set_title('Training Losses (Smoothed)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')  # Log scale for losses
    
    # 7. Score Statistics
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(episode_nums, score_means, label='Mean', linewidth=2)
    ax7.plot(episode_nums, score_mins, label='Min', linewidth=1, alpha=0.7)
    ax7.plot(episode_nums, score_maxs, label='Max', linewidth=1, alpha=0.7)
    score_stds = [ep.get('episode_score_std', 0) for ep in episodes]
    ax7.fill_between(episode_nums, 
                     np.array(score_means) - np.array(score_stds),
                     np.array(score_means) + np.array(score_stds),
                     alpha=0.2, label='±1 Std Dev')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Score (points)')
    ax7.set_title('Score Statistics (Mean, Min, Max, Std)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Policy and Value Losses (separate)
    ax8 = fig.add_subplot(gs[2, 1])
    ax6.plot(episode_nums, smooth_data(policy_losses, smooth_window), 
             label='Policy Loss', linewidth=2, color='blue')
    ax6.plot(episode_nums, smooth_data(value_losses, smooth_window), 
             label='Value Loss', linewidth=2, color='red')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Loss')
    ax6.set_title('Policy vs Value Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 9. Average Survival Time (Rolling Average)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(episode_nums, avg_survival_times, linewidth=2, color='purple', label='Avg Survival Time (100 ep window)')
    ax9.plot(episode_nums, length_means, linewidth=1.5, color='orange', alpha=0.7, label='Mean Episode Length')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Survival Time (steps)')
    ax9.set_title('Average Survival Time (Key Learning Metric)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Clip Fraction and KL Divergence
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(episode_nums, smooth_data(clip_fractions, smooth_window), 
             label='Clip Fraction', linewidth=2, color='orange')
    ax10.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='0.2 threshold')
    ax10.set_xlabel('Episode')
    ax10.set_ylabel('Clip Fraction')
    ax10.set_title('Clip Fraction (Fraction of Clipped Ratios)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. KL Divergence
    ax11 = fig.add_subplot(gs[3, 1])
    ax9.plot(episode_nums, smooth_data(entropies, smooth_window), 
             label='Entropy', linewidth=2, color='cyan')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Entropy')
    ax9.set_title('Policy Entropy (Exploration)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    ax11.plot(episode_nums, smooth_data(kl_divergences, smooth_window), 
             label='KL Divergence', linewidth=2, color='magenta')
    ax11.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='0.01 threshold')
    ax11.set_xlabel('Episode')
    ax11.set_ylabel('KL Divergence')
    ax11.set_title('Approximate KL Divergence')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Entropy
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(episode_nums, smooth_data(entropies, smooth_window), 
             label='Entropy', linewidth=2, color='cyan')
    ax12.set_xlabel('Episode')
    ax12.set_ylabel('Entropy')
    ax12.set_title('Policy Entropy (Exploration)')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # 13. Explained Variance
    ax13 = fig.add_subplot(gs[4, 0])
    ax13.plot(episode_nums, smooth_data(explained_variances, smooth_window), 
              label='Explained Variance', linewidth=2, color='darkgreen')
    ax13.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='0 threshold')
    ax13.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='1.0 (perfect)')
    ax13.set_xlabel('Episode')
    ax13.set_ylabel('Explained Variance')
    ax13.set_title('Explained Variance (Value Function Quality)')
    ax13.set_ylim([-0.5, 1.1])
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    
    # 14. Learning Rate
    ax14 = fig.add_subplot(gs[4, 1])
    ax14.plot(episode_nums, learning_rates, label='Learning Rate', linewidth=2, color='brown')
    ax14.set_xlabel('Episode')
    ax14.set_ylabel('Learning Rate')
    ax14.set_title('Learning Rate Schedule')
    ax14.legend()
    ax14.grid(True, alpha=0.3)
    ax14.set_yscale('log')  # Log scale for learning rate
    
    # 15. Reward vs Timestep
    ax15 = fig.add_subplot(gs[4, 2])
    ax15.plot(timesteps, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax15.plot(timesteps, smooth_data(episode_rewards, smooth_window), 
              color='blue', linewidth=2, label='Smoothed')
    ax15.set_xlabel('Timestep')
    ax15.set_ylabel('Reward')
    ax15.set_title('Reward vs Timestep')
    ax15.legend()
    ax15.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('PPO Training Metrics Visualization', fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_metrics_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(metrics_data: dict):
    """Print summary statistics from metrics."""
    episodes = metrics_data.get('episodes', [])
    summary = metrics_data.get('summary', {})
    
    if not episodes:
        print("No episode data found!")
        return
    
    print("\n" + "="*70)
    print("TRAINING METRICS SUMMARY")
    print("="*70)
    
    if summary:
        print(f"\nTotal Timesteps: {summary.get('total_timesteps', 'N/A'):,}")
        print(f"Total Episodes: {summary.get('total_episodes', 'N/A'):,}")
        print(f"Final Reward Mean: {summary.get('final_reward_mean', 0):.2f} ± {summary.get('final_reward_std', 0):.2f}")
        print(f"Final Survival Time (Episode Length): {summary.get('final_episode_length_mean', 0):.2f} ± {summary.get('final_episode_length_std', 0):.2f} steps")
        print(f"Best Survival Time: {summary.get('final_episode_length_max', 0)} steps")
        print(f"Final Game Score: {summary.get('final_score_mean', 0):.1f} ± {summary.get('final_score_std', 0):.1f} points")
        print(f"Best Game Score: {summary.get('final_score_max', 0)} points")
    else:
        # Calculate from episodes if summary not available
        final_rewards = [ep['episode_reward'] for ep in episodes]
        final_lengths = [ep['episode_length'] for ep in episodes]
        final_successes = [ep['episode_success'] for ep in episodes]
        
        print(f"\nTotal Episodes: {len(episodes):,}")
        if final_rewards:
            print(f"Final Reward Mean: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
            print(f"Best Reward: {np.max(final_rewards):.2f}")
            print(f"Worst Reward: {np.min(final_rewards):.2f}")
        if final_lengths:
            print(f"Final Survival Time (Episode Length): {np.mean(final_lengths):.2f} ± {np.std(final_lengths):.2f} steps")
            print(f"Best Survival Time: {np.max(final_lengths)} steps")
        final_scores = [ep.get('episode_score', 0) for ep in episodes]
        if final_scores:
            print(f"Final Game Score: {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f} points")
            print(f"Best Game Score: {np.max(final_scores)} points")
    
    # Last episode metrics
    if episodes:
        last_ep = episodes[-1]
        print(f"\nLast Episode Metrics:")
        print(f"  Episode: {last_ep.get('episode', 'N/A')}")
        print(f"  Reward: {last_ep.get('episode_reward', 0):.2f}")
        print(f"  Survival Time (Length): {last_ep.get('episode_length', 0)} steps")
        print(f"  Avg Survival Time: {last_ep.get('avg_survival_time', last_ep.get('episode_length_mean', 0)):.1f} steps")
        print(f"  Game Score: {last_ep.get('episode_score', 0)} points")
        print(f"  Avg Score: {last_ep.get('avg_score', last_ep.get('episode_score_mean', 0)):.1f} points")
        print(f"  Policy Loss: {last_ep.get('policy_loss', 0):.4f}")
        print(f"  Value Loss: {last_ep.get('value_loss', 0):.4f}")
        print(f"  Explained Variance: {last_ep.get('explained_variance', 0):.4f}")
    
    print("="*70 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize PPO training metrics from JSON file"
    )
    parser.add_argument(
        'metrics_file',
        type=str,
        help='Path to metrics JSON file (e.g., checkpoints/player_ppo/metrics/training_metrics_*.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save visualization (if not provided, displays plot)'
    )
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=100,
        help='Window size for smoothing curves (default: 100)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary statistics, do not create plots'
    )
    
    args = parser.parse_args()
    
    # Load metrics
    print(f"Loading metrics from: {args.metrics_file}")
    metrics_data = load_metrics(args.metrics_file)
    
    # Print summary
    print_summary(metrics_data)
    
    # Create visualizations
    if not args.summary_only:
        print("Creating visualizations...")
        create_visualizations(metrics_data, args.output_dir, args.smooth_window)
        print("Done!")


if __name__ == "__main__":
    main()