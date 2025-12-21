"""
Test Script for Trained PPO Agent

Load and test a trained PPO agent playing the zombie shooter game.
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.ppo_agent import PPOAgent, ZombieShooterEnv, PPOConfig


def test_agent(model_path: str, num_episodes: int = 10, render: bool = True, deterministic: bool = True):
    """
    Test a trained PPO agent.
    
    Args:
        model_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: If True, use deterministic policy (no sampling)
    """
    # Load config and create environment
    print(f"Loading model from: {model_path}")
    
    config = PPOConfig()
    config.headless = not render
    
    env = ZombieShooterEnv(config, headless=not render)
    agent = PPOAgent(config)
    
    # Load trained model
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"\nTesting agent for {num_episodes} episodes...")
    print(f"Deterministic policy: {deterministic}")
    print(f"Rendering: {render}\n")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    episode_kills = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        while not done:
            # Select action (deterministic or stochastic)
            action, _, _ = agent.select_action(state, deterministic=deterministic)
            
            # Take step
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Render
            if render:
                env.render()
            
            # Print progress every 100 steps
            if episode_length % 100 == 0:
                print(f"  Steps: {episode_length}, Reward: {episode_reward:.2f}, " +
                      f"Kills: {info.get('zombies_killed', 0)}, Health: {env.player['health']}")
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_kills.append(info.get('zombies_killed', 0))
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Zombies Killed: {info.get('zombies_killed', 0)}")
        print(f"  Final Health: {env.player['health']}")
        
        if 'death' in info:
            print(f"  Result: DIED")
        elif 'timeout' in info:
            print(f"  Result: TIMEOUT")
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Kills: {np.mean(episode_kills):.1f} ± {np.std(episode_kills):.1f}")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Reward: {np.min(episode_rewards):.2f}")
    print(f"{'='*50}\n")
    
    env.close()


def main():
    """Main function for testing agent."""
    parser = argparse.ArgumentParser(description="Test trained PPO agent")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default is deterministic)')
    
    args = parser.parse_args()
    
    test_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
