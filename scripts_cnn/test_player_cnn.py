"""
Test Script for CNN-based Player Agent

Test a trained CNN-based player agent.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents_cnn.player_cnn.agent import CNNPlayerAgent
from agents_cnn.player_cnn.config import CNNPlayerConfig
from agents_cnn.player_cnn.env_wrapper import CNNEnvWrapper
from agents.ppo_agent.config import PPOConfig


def test_cnn_player(model_path: str, num_episodes: int = 10, render: bool = True, deterministic: bool = True):
    """
    Test a trained CNN-based player agent.
    
    Args:
        model_path: Path to trained CNN model checkpoint
        num_episodes: Number of episodes to test
        render: Whether to render the game
        deterministic: Whether to use deterministic policy
    """
    print("=" * 60)
    print("Testing CNN-based Player Agent")
    print("=" * 60)
    
    # Create config
    config = CNNPlayerConfig()
    config.headless = not render
    
    # Create PPO config for environment
    ppo_config = PPOConfig()
    
    # Create CNN environment wrapper
    print("Creating CNN environment wrapper...")
    env = CNNEnvWrapper(ppo_config, headless=not render)
    
    # Create CNN agent
    print("Creating CNN agent...")
    agent = CNNPlayerAgent(config)
    
    # Load model
    print(f"Loading model from {model_path}...")
    agent.load(model_path)
    
    print(f"\nTesting for {num_episodes} episodes...")
    print(f"Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(state, deterministic=deterministic)
            
            # Take step
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info.get('score', 0))
        
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Length: {episode_length} | "
              f"Score: {info.get('score', 0)} | "
              f"Kills: {info.get('zombies_killed', 0)} | "
              f"Wave: {info.get('wave', 1)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Average Length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"Average Score: {sum(episode_scores) / len(episode_scores):.1f}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Best Score: {max(episode_scores)}")
    print("=" * 60)
    
    env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test CNN-based Player Agent")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained CNN model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default is deterministic)')
    
    args = parser.parse_args()
    
    test_cnn_player(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
