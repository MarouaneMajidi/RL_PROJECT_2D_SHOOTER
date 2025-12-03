"""
Training Script for PPO Agent

Train a PPO agent to play the zombie shooter game.
"""

import os
import sys
import time
import numpy as np
import torch
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo_agent import PPOAgent, ZombieShooterEnv, PPOConfig


class Logger:
    """Simple logger for training statistics."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write("timestep,episode,episode_reward,episode_length,zombies_killed,wave," +
                   "policy_loss,value_loss,entropy_loss,total_loss,clip_fraction,approx_kl," +
                   "fps,elapsed_time\n")
    
    def log(self, data: dict):
        """Log training data."""
        with open(self.log_file, 'a') as f:
            line = ','.join([str(data.get(key, '')) for key in [
                'timestep', 'episode', 'episode_reward', 'episode_length', 'zombies_killed', 'wave',
                'policy_loss', 'value_loss', 'entropy_loss', 'total_loss', 'clip_fraction',
                'approx_kl', 'fps', 'elapsed_time'
            ]])
            f.write(line + '\n')
    
    def print_log(self, data: dict):
        """Print training statistics to console."""
        print(f"\n{'='*70}")
        print(f"Timestep: {data.get('timestep', 0):,} | Episode: {data.get('episode', 0)}")
        print(f"Episode Reward: {data.get('episode_reward', 0):.2f} | Length: {data.get('episode_length', 0)}")
        print(f"Zombies Killed: {data.get('zombies_killed', 0)} | Wave: {data.get('wave', 1)}")
        
        if 'policy_loss' in data:
            print(f"\nTraining Stats:")
            print(f"  Policy Loss: {data.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {data.get('value_loss', 0):.4f}")
            print(f"  Entropy Loss: {data.get('entropy_loss', 0):.4f}")
            print(f"  Total Loss: {data.get('total_loss', 0):.4f}")
            print(f"  Clip Fraction: {data.get('clip_fraction', 0):.4f}")
            print(f"  Approx KL: {data.get('approx_kl', 0):.4f}")
        
        print(f"\nFPS: {data.get('fps', 0):.1f} | Elapsed: {data.get('elapsed_time', 0):.1f}s")
        print(f"{'='*70}")


def train_ppo(
    config: Optional[PPOConfig] = None,
    checkpoint_path: Optional[str] = None,
    render: bool = False
):
    """
    Train a PPO agent on the zombie shooter environment.
    
    Args:
        config: PPOConfig object (uses default if None)
        checkpoint_path: Path to load checkpoint from (if continuing training)
        render: Whether to render the environment during training
    """
    # Use default config if not provided
    if config is None:
        config = PPOConfig()
    
    # Create environment
    print("Creating environment...")
    env = ZombieShooterEnv(config, headless=not render)
    
    # Create agent
    print("Creating PPO agent...")
    agent = PPOAgent(config)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        agent.load(checkpoint_path)
    
    # Create logger
    logger = Logger()
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Rollout steps: {config.n_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs per update: {config.n_epochs}")
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    best_reward = -float('inf')
    
    start_time = time.time()
    last_log_time = start_time
    timesteps_since_log = 0
    
    while agent.total_timesteps < config.total_timesteps:
        # Collect rollout
        for step in range(config.n_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            timesteps_since_log += 1
            
            # Render if enabled
            if render:
                env.render()
            
            state = next_state
            
            # Handle episode end
            if done:
                episode_count += 1
                
                # Prepare log data
                log_data = {
                    'timestep': agent.total_timesteps,
                    'episode': episode_count,
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'zombies_killed': info.get('zombies_killed', 0),
                    'wave': info.get('wave', 1),
                    'fps': timesteps_since_log / (time.time() - last_log_time),
                    'elapsed_time': time.time() - start_time
                }
                
                # Log and print
                logger.log(log_data)
                logger.print_log(log_data)
                
                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save(config.best_model_path)
                    print(f"New best reward: {best_reward:.2f} - Model saved!")
                
                # Reset episode tracking
                episode_reward = 0
                episode_length = 0
                last_log_time = time.time()
                timesteps_since_log = 0
                
                # Reset environment
                state = env.reset()
        
        # Perform PPO update
        last_value = agent.get_value(state)
        update_stats = agent.update(last_value)
        
        # Log update statistics
        current_time = time.time()
        log_data = {
            'timestep': agent.total_timesteps,
            'policy_loss': update_stats['policy_loss'],
            'value_loss': update_stats['value_loss'],
            'entropy_loss': update_stats['entropy_loss'],
            'total_loss': update_stats['total_loss'],
            'clip_fraction': update_stats['clip_fraction'],
            'approx_kl': update_stats['approx_kl'],
            'elapsed_time': current_time - start_time
        }
        
        print(f"\n--- Update {update_stats['num_updates']} ---")
        print(f"Timesteps: {agent.total_timesteps:,} / {config.total_timesteps:,}")
        print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"Value Loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {update_stats['entropy_loss']:.4f}")
        print(f"Clip Fraction: {update_stats['clip_fraction']:.4f}")
        
        # Save checkpoint periodically
        if agent.total_timesteps % config.save_interval < config.n_steps:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"checkpoint_{agent.total_timesteps}.pth"
            )
            agent.save(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    agent.save(final_path)
    
    # Close environment
    env.close()
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {config.best_model_path}")
    print("="*70)


def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent on zombie shooter")
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total timesteps to train (default: 1,000,000)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to continue training from')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per rollout (default: 2048)')
    
    args = parser.parse_args()
    
    # Create custom config
    config = PPOConfig()
    config.total_timesteps = args.timesteps
    config.device = args.device
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.n_steps = args.n_steps
    config.headless = not args.render
    
    # Train
    train_ppo(config=config, checkpoint_path=args.checkpoint, render=args.render)


if __name__ == "__main__":
    main()
