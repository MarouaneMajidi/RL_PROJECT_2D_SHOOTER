"""
Test DDA Agent with both manual and agent player modes.

Usage:
    python test_dda.py --mode manual    # You play
    python test_dda.py --mode agent    # PPO agent plays
"""

import os
import sys
import argparse
import pygame
from pygame.locals import *

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from agents.dda_agent import DDAAgent, DDAConfig, DDAStateExtractor, DifficultyManager
from agents.ppo_agent import PPOAgent, PPOConfig, ZombieShooterEnv

# Colors for UI
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def test_dda_with_agent(dda_model_path: str, player_model_path: str):
    """
    Test DDA agent with PPO player agent.
    
    Args:
        dda_model_path: Path to DDA model
        player_model_path: Path to player PPO model
    """
    print("="*60)
    print("DDA TEST MODE: PPO Agent Playing")
    print("="*60)
    
    # Load DDA agent
    print("Loading DDA agent...")
    dda_config = DDAConfig()
    dda_agent = DDAAgent(dda_config)
    
    try:
        dda_agent.load(dda_model_path)
        print(f"✓ DDA agent loaded from {dda_model_path}")
    except FileNotFoundError:
        print(f"✗ Error: DDA model not found at {dda_model_path}")
        print("Please train the DDA agent first: python train_dda.py")
        return
    
    # Load player agent
    print("Loading player agent...")
    ppo_config = PPOConfig()
    player_agent = PPOAgent(ppo_config)
    
    try:
        player_agent.load(player_model_path)
        print(f"✓ Player agent loaded from {player_model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Player model not found at {player_model_path}")
        print("Please train the player agent first: python train_ppo.py")
        return
    
    # Create game environment
    print("Creating game environment...")
    game_env = ZombieShooterEnv(ppo_config, headless=False)
    
    # Get base parameters (for difficulty calculation)
    base_params = {
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
    
    # Store base_params for difficulty calculation
    _base_params_for_difficulty = base_params
    
    # Initialize DDA components
    state_extractor = DDAStateExtractor(window_size=600, fps=60)
    difficulty_manager = DifficultyManager(base_params)
    
    # Apply initial parameters
    params = difficulty_manager.get_current_params()
    _apply_params_to_env(game_env, params)
    
    # Game loop
    print("\nStarting game...")
    print("The DDA agent will adjust difficulty every 5 seconds.")
    print("Press ESC to quit\n")
    
    state = game_env.reset()
    state_extractor.reset()
    difficulty_manager.reset()
    
    dda_action_timer = 0
    dda_action_interval = 300  # 5 seconds
    episode_count = 0
    last_dda_action = "None"
    previous_dda_action = "None"  # Previous decision
    current_params = difficulty_manager.get_current_params()
    
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_r:
                    # Reset
                    state = game_env.reset()
                    state_extractor.reset()
                    difficulty_manager.reset()
                    params = difficulty_manager.get_current_params()
                    _apply_params_to_env(game_env, params)
                    episode_count += 1
                    print(f"\nEpisode {episode_count} - Reset!")
        
        # DDA action
        dda_action_timer += 1
        if dda_action_timer >= dda_action_interval:
            dda_action_timer = 0
            
            # Extract DDA state
            game_state_info = {
                'player_health': game_env.player.get('health', 100),
                'zombies_alive': len(game_env.zombies),
                'max_zombies': 20,
                'machinegun_ammo': game_env.player.get('machinegun_ammo', 0),
                'max_ammo': 100,
            }
            
            dda_state = state_extractor.extract_state(game_state_info)
            
            # Get DDA action
            action, _, _ = dda_agent.select_action(dda_state, deterministic=True)
            
            # Apply action
            difficulty_manager.apply_action(action)
            current_params = difficulty_manager.get_current_params()
            _apply_params_to_env(game_env, current_params)
            
            action_names = ["Do Nothing", "Slightly Easier", "Much Easier", "Slightly Harder", "Much Harder"]
            # Update previous action before setting new one
            if last_dda_action != "None":
                previous_dda_action = last_dda_action
            last_dda_action = action_names[action]
            
            print(f"\n{'='*60}")
            print(f"DDA Action: {last_dda_action}")
            print(f"Player Health: {game_state_info['player_health']:.1f}")
            print(f"Zombies Alive: {game_state_info['zombies_alive']}")
            print(f"Difficulty Params:")
            print(f"  Zombie Speed: {current_params['ZOMBIE_SPEED']:.2f}")
            print(f"  Spawn Rate: {current_params['SPAWN_RATE']:.0f}")
            print(f"  Zombie Health: {current_params['ZOMBIE_NORMAL_HEALTH']:.0f}")
            print(f"  Player Speed: {current_params['PLAYER_SPEED']:.2f}")
            print(f"{'='*60}\n")
        
        # Get player action from PPO agent
        player_action, _, _ = player_agent.select_action(state, deterministic=False)
        
        # Step game
        next_state, reward, done, info = game_env.step(player_action)
        
        # Update metrics
        last_health = state_extractor.last_player_health if hasattr(state_extractor, 'last_player_health') else 100
        current_health = game_env.player.get('health', 100)
        damage_taken = max(0, last_health - current_health)
        
        # Track kills
        zombie_killed = False
        if 'zombies_killed' in info:
            zombie_killed = True
        
        state_extractor.update({
            'player_health': current_health,
            'damage_taken': damage_taken,
            'damage_dealt': 0,  # Simplified
            'zombie_killed': zombie_killed,
            'shot_fired': player_action % 2 == 1,  # Shoot action
            'shot_hit': zombie_killed,
            'pickup_collected': False,  # Simplified
        })
        
        state = next_state
        
        # Render game first
        game_env.render()
        
        # Draw DDA overlay on top (render() already flipped, so we draw and flip again)
        _draw_dda_overlay(game_env.screen, last_dda_action, previous_dda_action, current_params, _base_params_for_difficulty)
        pygame.display.flip()
        
        # Handle episode end
        if done:
            print(f"\nEpisode ended! Health: {current_health:.1f}, Kills: {info.get('zombies_killed', 0)}")
            state = game_env.reset()
            state_extractor.reset()
            difficulty_manager.reset()
            params = difficulty_manager.get_current_params()
            _apply_params_to_env(game_env, params)
            episode_count += 1
    
    print("\nTest completed!")
    game_env.close()


def _calculate_difficulty_level(current_params, base_params):
    """
    Calculate overall difficulty level based on current parameters.
    
    Returns:
        (difficulty_level_string, color)
    """
    # Calculate difficulty score (0.0 = very easy, 1.0 = very hard)
    # Based on how parameters differ from base
    
    # Enemy difficulty factors (higher = harder)
    zombie_speed_factor = (current_params['ZOMBIE_SPEED'] - base_params['ZOMBIE_SPEED']) / base_params['ZOMBIE_SPEED']
    spawn_rate_factor = (base_params['SPAWN_RATE'] - current_params['SPAWN_RATE']) / base_params['SPAWN_RATE']  # Lower spawn = harder
    zombie_health_factor = (current_params['ZOMBIE_NORMAL_HEALTH'] - base_params['ZOMBIE_NORMAL_HEALTH']) / base_params['ZOMBIE_NORMAL_HEALTH']
    
    # Player power factors (higher = easier, so we invert)
    player_speed_factor = -(current_params['PLAYER_SPEED'] - base_params['PLAYER_SPEED']) / base_params['PLAYER_SPEED']
    weapon_damage_factor = -(current_params['PISTOL_DAMAGE'] - base_params['PISTOL_DAMAGE']) / base_params['PISTOL_DAMAGE']
    
    # Resource factors (higher = easier, so we invert)
    health_drop_factor = -(current_params['HEALTH_PICKUP_DROP_PROBABILITY'] - base_params['HEALTH_PICKUP_DROP_PROBABILITY']) / base_params['HEALTH_PICKUP_DROP_PROBABILITY']
    
    # Average difficulty score
    difficulty_score = (zombie_speed_factor + spawn_rate_factor + zombie_health_factor + 
                        player_speed_factor + weapon_damage_factor + health_drop_factor) / 6.0
    
    # Normalize to [0, 1] range
    difficulty_score = max(-1.0, min(1.0, difficulty_score))
    normalized_score = (difficulty_score + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
    
    # Map to difficulty levels
    if normalized_score < 0.2:
        return "VERY EASY", GREEN
    elif normalized_score < 0.4:
        return "EASY", (100, 255, 100)  # Light green
    elif normalized_score < 0.6:
        return "NORMAL", YELLOW
    elif normalized_score < 0.8:
        return "HARD", (255, 150, 0)  # Orange
    else:
        return "VERY HARD", RED


def _draw_dda_overlay(screen, last_dda_action, previous_dda_action, params, base_params):
    """Draw DDA actions (fixed, always visible, updates when DDA acts)."""
    font = pygame.font.SysFont(None, 36)
    font_small = pygame.font.SysFont(None, 24)
    
    SCREEN_WIDTH = 1000
    x_offset = SCREEN_WIDTH - 280
    y_start = 30
    
    # Determine color based on action
    def get_action_color(action):
        if "Easier" in action:
            return GREEN
        elif "Harder" in action:
            return RED
        else:
            return WHITE
    
    # Title
    screen.blit(font.render("DDA DECISIONS", True, YELLOW), (x_offset, y_start))
    
    # Previous decision (smaller, grayed out)
    prev_color = get_action_color(previous_dda_action)
    prev_color_grayed = tuple(c // 2 for c in prev_color)  # Darker version
    screen.blit(font_small.render("Previous:", True, (128, 128, 128)), (x_offset, y_start + 40))
    screen.blit(font_small.render(previous_dda_action, True, prev_color_grayed), (x_offset, y_start + 60))
    
    # Current decision (bigger, bright)
    current_color = get_action_color(last_dda_action)
    screen.blit(font_small.render("Current:", True, WHITE), (x_offset, y_start + 85))
    screen.blit(font.render(last_dda_action, True, current_color), (x_offset, y_start + 105))


def _apply_params_to_env(env, params):
    """Apply difficulty parameters to game environment."""
    if hasattr(env, 'ZOMBIE_SPEED'):
        env.ZOMBIE_SPEED = params['ZOMBIE_SPEED']
    if hasattr(env, 'SPAWN_RATE'):
        env.SPAWN_RATE = int(params['SPAWN_RATE'])
        env.spawn_rate = int(params['SPAWN_RATE'])
    if hasattr(env, 'ZOMBIE_ATTACK_COOLDOWN'):
        env.ZOMBIE_ATTACK_COOLDOWN = int(params['ZOMBIE_ATTACK_COOLDOWN'])
    if hasattr(env, 'ZOMBIE_NORMAL_HEALTH'):
        env.ZOMBIE_NORMAL_HEALTH = int(params['ZOMBIE_NORMAL_HEALTH'])
    if hasattr(env, 'ZOMBIE_STRONG_HEALTH'):
        env.ZOMBIE_STRONG_HEALTH = int(params['ZOMBIE_STRONG_HEALTH'])
    if hasattr(env, 'PLAYER_SPEED'):
        env.PLAYER_SPEED = params['PLAYER_SPEED']
    if hasattr(env, 'PISTOL_DAMAGE'):
        env.PISTOL_DAMAGE = int(params['PISTOL_DAMAGE'])
    if hasattr(env, 'MACHINEGUN_DAMAGE'):
        env.MACHINEGUN_DAMAGE = int(params['MACHINEGUN_DAMAGE'])
    if hasattr(env, 'HEALTH_PICKUP_DROP_PROBABILITY'):
        env.HEALTH_PICKUP_DROP_PROBABILITY = params['HEALTH_PICKUP_DROP_PROBABILITY']
    if hasattr(env, 'MACHINEGUN_PICKUP_DROP_PROBABILITY'):
        env.MACHINEGUN_PICKUP_DROP_PROBABILITY = params['MACHINEGUN_PICKUP_DROP_PROBABILITY']
    if hasattr(env, 'PISTOL_COOLDOWN'):
        env.PISTOL_COOLDOWN = int(params['PISTOL_COOLDOWN'])
    if hasattr(env, 'MACHINEGUN_COOLDOWN'):
        env.MACHINEGUN_COOLDOWN = int(params['MACHINEGUN_COOLDOWN'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DDA Agent")
    parser.add_argument('--mode', type=str, choices=['manual', 'agent'], default='manual',
                       help='Test mode: manual (you play) or agent (PPO agent plays)')
    parser.add_argument('--dda-model', type=str, default='checkpoints/dda/best_dda_model.pth',
                       help='Path to DDA model')
    parser.add_argument('--player-model', type=str, default='checkpoints/best_model.pth',
                       help='Path to player PPO model (for agent mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        # Run manual mode (play_with_dda.py)
        print("Starting manual mode (you play)...")
        print("Run: python play_with_dda.py --model", args.dda_model)
        import subprocess
        subprocess.run([sys.executable, 'play_with_dda.py', '--model', args.dda_model])
    else:
        # Run agent mode
        test_dda_with_agent(args.dda_model, args.player_model)

