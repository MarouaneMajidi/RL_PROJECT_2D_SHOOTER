"""
Training Script for DDA Agent

Trains a reinforcement learning agent to dynamically adjust game difficulty.
"""

import os
import sys
import time
import numpy as np
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.ppo_agent import PPOAgent, PPOConfig, ZombieShooterEnv
from agents.dda_agent import DDAAgent, DDAConfig, DDAEnvironment, DifficultyManager
from utils.dda_training_metrics import DDATrainingMetricsTracker


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


def train_dda(
    player_model_path: str = os.path.join(project_root, "checkpoints/best_model.pth"),
    dda_config: Optional[DDAConfig] = None,
    checkpoint_path: Optional[str] = None,
    render: bool = False,
    max_episodes: Optional[int] = None
):
    """
    Train DDA agent.
    
    Args:
        player_model_path: Path to trained player agent model
        dda_config: DDAConfig object (uses default if None)
        checkpoint_path: Path to DDA checkpoint to resume from (optional)
        render: Whether to render during training
        max_episodes: Maximum number of episodes to train (if None, uses timesteps from config)
    """
    print("=" * 60)
    print("DDA Agent Training")
    print("=" * 60)
    
    # Initialize configs
    if dda_config is None:
        dda_config = DDAConfig()
    
    ppo_config = PPOConfig()
    
    # Create game environment
    print("Creating game environment...")
    game_env = ZombieShooterEnv(ppo_config, headless=not render)
    
    # Load trained player agent
    print(f"Loading player agent from {player_model_path}...")
    player_agent = PPOAgent(ppo_config)
    try:
        player_agent.load(player_model_path)
        print("Player agent loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Player agent not found at {player_model_path}")
        print("Training DDA without player agent (will use random actions)")
        player_agent = None
    
    # Get base parameters
    base_params = get_base_params()
    
    # Create DDA environment
    print("Creating DDA environment...")
    dda_env = DDAEnvironment(
        game_env=game_env,
        base_params=base_params,
        action_interval=dda_config.action_interval,
        fps=60
    )
    
    # Set player agent in DDA environment
    if player_agent is not None:
        dda_env.set_player_agent(player_agent)
    
    # Create DDA agent
    print("Creating DDA agent...")
    dda_agent = DDAAgent(dda_config)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading DDA checkpoint from {checkpoint_path}...")
        dda_agent.load(checkpoint_path)
    
    # Create checkpoint directory
    os.makedirs(dda_config.checkpoint_dir, exist_ok=True)
    
    # Create metrics tracker
    metrics_tracker = DDATrainingMetricsTracker(save_dir=os.path.join(dda_config.checkpoint_dir, "metrics"))
    
    # Training loop
    print("\nStarting training...")
    if max_episodes is not None:
        print(f"Max episodes: {max_episodes}")
    else:
    print(f"Total timesteps: {dda_config.total_timesteps}")
    print(f"Action interval: {dda_config.action_interval} frames ({dda_config.action_interval/60:.1f} seconds)")
    print(f"Metrics will be saved to: {metrics_tracker.metrics_file}")
    print("-" * 60)
    
    episode_count = 0
    best_reward = float('-inf')
    start_time = time.time()
    
    # Initialize episode tracking
    metrics_tracker.start_episode()
    last_update_metrics = None
    
    # Training loop condition: use episodes if specified, otherwise use timesteps
    should_continue = True
    while should_continue:
        # Reset environment
        state = dda_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Collect rollout
        for step in range(dda_config.n_steps):
            # Select DDA action
            action, log_prob, value = dda_agent.select_action(state)
            
            # Take step in DDA environment
            next_state, reward, done, info = dda_env.step(action)
            
            # Store transition
            dda_agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update metrics tracker with step
            metrics_tracker.update_episode_step(reward, info, dda_action=action)
            
            state = next_state
            
            # Handle episode end
            if done:
                episode_count += 1
                
                # Get current learning rate from optimizer
                current_lr = dda_agent.optimizer.param_groups[0]['lr']
                
                # End episode in metrics tracker (saves metrics to JSON)
                metrics_tracker.end_episode(
                    timestep=dda_agent.total_timesteps,
                    update_metrics=last_update_metrics,
                    learning_rate=current_lr
                )
                
                # Log episode
                if episode_count % dda_config.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = dda_agent.total_timesteps / elapsed if elapsed > 0 else 0
                    
                    print(f"Episode {episode_count} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Length: {episode_length} | "
                          f"Health: {info.get('player_health', 0):.1f} | "
                          f"Zombies: {info.get('zombies_alive', 0)} | "
                          f"Timesteps: {dda_agent.total_timesteps} | "
                          f"FPS: {fps:.1f}")
                
                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    dda_agent.save(dda_config.best_model_path)
                    print(f"New best reward: {best_reward:.2f} - Model saved!")
                
                # Start new episode in metrics tracker
                metrics_tracker.start_episode()
                
                # Check if we've reached max episodes
                if max_episodes is not None and episode_count >= max_episodes:
                    should_continue = False
                    break
                
                # Reset for next episode
                break
        
        # Perform PPO update
        if dda_agent.rollout_buffer.pos > 0:
            # Get value for last state
            last_value = dda_agent.get_value(state) if not done else 0.0
            
            # Update agent
            update_stats = dda_agent.update(last_value)
            
            # Get current learning rate from optimizer
            current_lr = dda_agent.optimizer.param_groups[0]['lr']
            
            # Record update metrics in tracker
            metrics_tracker.record_update(update_stats, current_lr)
            last_update_metrics = update_stats
            
            # Log training stats
            if dda_agent.num_updates % dda_config.log_interval == 0:
                print(f"Update {dda_agent.num_updates} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy_loss']:.4f} | "
                      f"Clip Fraction: {update_stats['clip_fraction']:.4f}")
                if 'explained_variance' in update_stats:
                    print(f"  Explained Variance: {update_stats['explained_variance']:.4f}")
        
        # Save checkpoint periodically
        if dda_agent.total_timesteps % dda_config.save_interval == 0:
            checkpoint_path = os.path.join(
                dda_config.checkpoint_dir,
                f"checkpoint_{dda_agent.total_timesteps}.pth"
            )
            dda_agent.save(checkpoint_path)
            print(f"Checkpoint saved at {dda_agent.total_timesteps} timesteps")
        
        # Check if we should continue (for timestep-based training)
        if max_episodes is None and dda_agent.total_timesteps >= dda_config.total_timesteps:
            should_continue = False
    
    # Finalize metrics tracking (saves final summary)
    metrics_tracker.finalize(total_timesteps=dda_agent.total_timesteps)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total episodes: {episode_count}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final model saved to: {dda_config.best_model_path}")
    print(f"Metrics saved to: {metrics_tracker.metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DDA Agent")
    parser.add_argument('--player-model', type=str, default=os.path.join(project_root, 'checkpoints/player_ppo/best_model.pth'),
                       help='Path to trained player agent model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to DDA checkpoint to resume from')
    parser.add_argument('--render', action='store_true',
                       help='Render during training')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Maximum number of episodes to train (if specified, overrides timesteps)')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Maximum number of timesteps to train (if episodes not specified)')
    
    args = parser.parse_args()
    
    # Create config and override timesteps if specified
    dda_config = None
    if args.timesteps is not None:
        from agents.dda_agent import DDAConfig
        dda_config = DDAConfig()
        dda_config.total_timesteps = args.timesteps
    
    train_dda(
        player_model_path=args.player_model,
        dda_config=dda_config,
        checkpoint_path=args.checkpoint,
        render=args.render,
        max_episodes=args.episodes
    )

