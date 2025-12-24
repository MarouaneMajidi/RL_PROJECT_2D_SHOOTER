"""
Test Script for CNN-based DDA Agent

Test a trained CNN-based DDA agent.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents_cnn.dda_cnn.agent import CNNDDAAgent
from agents_cnn.dda_cnn.config import CNNDDAConfig
from agents_cnn.dda_cnn.env_wrapper import CNNDDAEnvWrapper
from agents.ppo_agent import PPOAgent, PPOConfig, ZombieShooterEnv
from agents.dda_agent import DDAEnvironment


def get_base_params():
    """Get base game parameters."""
    return {
        'ZOMBIE_SPEED': 2.0,
        'SPAWN_RATE': 60,
        'ZOMBIE_ATTACK_COOLDOWN': 9,
        'ZOMBIE_NORMAL_HEALTH': 30,
        'ZOMBIE_STRONG_HEALTH': 50,
        'PLAYER_SPEED': 5.0,
        'PISTOL_DAMAGE': 20,
        'MACHINEGUN_DAMAGE': 10,
        'HEALTH_PICKUP_DROP_PROBABILITY': 0.40,
        'MACHINEGUN_PICKUP_DROP_PROBABILITY': 0.15,
        'PISTOL_COOLDOWN': 20,
        'MACHINEGUN_COOLDOWN': 5,
    }


def test_cnn_dda(dda_model_path: str, player_model_path: str = None, 
                 num_episodes: int = 5, render: bool = True, deterministic: bool = True):
    """
    Test a trained CNN-based DDA agent.
    
    Args:
        dda_model_path: Path to trained CNN DDA model checkpoint
        player_model_path: Path to player agent model (optional)
        num_episodes: Number of episodes to test
        render: Whether to render the game
        deterministic: Whether to use deterministic policy
    """
    print("=" * 60)
    print("Testing CNN-based DDA Agent")
    print("=" * 60)
    
    # Create configs
    dda_config = CNNDDAConfig()
    dda_config.headless = not render
    
    ppo_config = PPOConfig()
    
    # Create game environment
    print("Creating game environment...")
    game_env = ZombieShooterEnv(ppo_config, headless=not render)
    
    # Load player agent if provided
    player_agent = None
    if player_model_path and os.path.exists(player_model_path):
        print(f"Loading player agent from {player_model_path}...")
        player_agent = PPOAgent(ppo_config)
        try:
            player_agent.load(player_model_path)
            print("Player agent loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load player agent: {e}")
            player_agent = None
    
    # Get base parameters
    base_params = get_base_params()
    
    # Create feature-based DDA environment
    print("Creating DDA environment...")
    dda_env_feature = DDAEnvironment(
        game_env=game_env,
        base_params=base_params,
        action_interval=dda_config.action_interval,
        fps=60
    )
    
    if player_agent:
        dda_env_feature.set_player_agent(player_agent)
    
    # Wrap with CNN wrapper
    print("Creating CNN environment wrapper...")
    dda_env = CNNDDAEnvWrapper(dda_env_feature)
    
    # Create CNN DDA agent
    print("Creating CNN DDA agent...")
    dda_agent = CNNDDAAgent(dda_config)
    
    # Load model
    print(f"Loading CNN DDA model from {dda_model_path}...")
    dda_agent.load(dda_model_path)
    
    print(f"\nTesting for {num_episodes} episodes...")
    print(f"Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state = dda_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select DDA action
            action, _, _ = dda_agent.select_action(state, deterministic=deterministic)
            
            # Take step
            state, reward, done, info = dda_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                dda_env_feature.game_env.render()
        
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Length: {episode_length} | "
              f"Player Health: {info.get('player_health', 0):.1f} | "
              f"Zombies Alive: {info.get('zombies_alive', 0)}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    dda_env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test CNN-based DDA Agent")
    parser.add_argument('--dda-model', type=str, required=True,
                       help='Path to trained CNN DDA model checkpoint')
    parser.add_argument('--player-model', type=str, default=None,
                       help='Path to player agent model (optional)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to test (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default is deterministic)')
    
    args = parser.parse_args()
    
    test_cnn_dda(
        dda_model_path=args.dda_model,
        player_model_path=args.player_model,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
