"""
Evaluation Script for DDA Agents

Runs 100 games per DDA agent and collects comprehensive metrics for comparison.
Supports both RL-based (feature) and CNN-based DDA agents.
"""

import os
import sys
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get actual project root (parent directory if script is in evaluate_find_best_dda)
if os.path.basename(script_dir) == 'evaluate_find_best_dda':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
sys.path.insert(0, project_root)

from agents.ppo_agent import PPOAgent, PPOConfig, ZombieShooterEnv
from agents.dda_agent import DDAAgent, DDAConfig, DDAEnvironment
from agents_cnn.dda_cnn.agent import CNNDDAAgent
from agents_cnn.dda_cnn.config import CNNDDAConfig
from agents_cnn.dda_cnn.env_wrapper import CNNDDAEnvWrapper

# Try to import tqdm for progress bars, but make it optional
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


# Action name mapping
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Slightly Easier",
    2: "Much Easier",
    3: "Slightly Harder",
    4: "Much Harder"
}


def get_base_params():
    """Get base difficulty parameters."""
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


def find_dda_models():
    """Find all DDA model files (both RL and CNN)."""
    models = []
    
    # Find RL-based DDA models
    rl_dda_dir = os.path.join(project_root, "checkpoints", "dda")
    if os.path.exists(rl_dda_dir):
        for filename in os.listdir(rl_dda_dir):
            if filename.endswith('.pth'):
                model_path = os.path.join(rl_dda_dir, filename)
                model_name = filename.replace('.pth', '')
                models.append({
                    'path': model_path,
                    'name': f"RL_{model_name}",
                    'type': 'rl'
                })
    
    # Find CNN-based DDA models
    cnn_dda_dir = os.path.join(project_root, "checkpoints_cnn", "dda_cnn")
    if os.path.exists(cnn_dda_dir):
        for filename in os.listdir(cnn_dda_dir):
            if filename.endswith('.pth'):
                model_path = os.path.join(cnn_dda_dir, filename)
                model_name = filename.replace('.pth', '')
                models.append({
                    'path': model_path,
                    'name': f"CNN_{model_name}",
                    'type': 'cnn'
                })
    
    return models


def is_cnn_model(model_path: str) -> bool:
    """Check if a model is CNN-based by trying to load it."""
    # Simple heuristic: CNN models are in checkpoints_cnn directory
    return "checkpoints_cnn" in model_path


def calculate_action_oscillations(actions: List[int]) -> int:
    """Calculate number of oscillations (direction changes) in action sequence."""
    if len(actions) < 3:
        return 0
    
    oscillations = 0
    for i in range(1, len(actions) - 1):
        prev_action = actions[i - 1]
        curr_action = actions[i]
        next_action = actions[i + 1]
        
        # Check if direction changed: easier -> harder -> easier or harder -> easier -> harder
        # Easier actions: 1, 2 (negative adjustment)
        # Harder actions: 3, 4 (positive adjustment)
        # Do nothing: 0 (neutral)
        
        prev_is_easier = prev_action in [1, 2]
        curr_is_harder = curr_action in [3, 4]
        next_is_easier = next_action in [1, 2]
        
        prev_is_harder = prev_action in [3, 4]
        curr_is_easier = curr_action in [1, 2]
        next_is_harder = next_action in [3, 4]
        
        if (prev_is_easier and curr_is_harder and next_is_easier) or \
           (prev_is_harder and curr_is_easier and next_is_harder):
            oscillations += 1
    
    return oscillations


def run_game_with_dda(dda_agent, dda_env, is_cnn: bool, game_id: int) -> Tuple[Dict, List[Dict]]:
    """
    Run a single game with DDA agent and collect metrics.
    
    Returns:
        game_metrics: Dictionary with per-game aggregated metrics
        interval_metrics: List of dictionaries with per-interval metrics
    """
    # Reset environment
    state = dda_env.reset()
    done = False
    
    # Per-interval tracking
    interval_metrics = []
    current_interval = 0
    
    # Per-game tracking
    health_history = []
    action_history = []
    action_counts = defaultdict(int)
    
    # Track metrics at interval boundaries
    interval_start_health = 100.0
    interval_start_zombies = 0
    interval_start_kills = 0
    interval_start_damage_taken = 0.0
    interval_start_damage_dealt = 0.0
    last_action = 0
    
    # Get game environment reference for direct access
    if hasattr(dda_env, 'game_env'):
        game_env = dda_env.game_env
    elif hasattr(dda_env, 'dda_env') and hasattr(dda_env.dda_env, 'game_env'):
        game_env = dda_env.dda_env.game_env
    else:
        game_env = None
    
    # Track when DDA actions occur
    # The DDA environment steps internally, so we need to track metrics differently
    # We'll step the environment and collect metrics at each step
    # DDA actions occur every action_interval frames (300 = 5 seconds)
    
    step_count = 0
    last_dda_action_step = -1
    
    while not done:
        # Get DDA action based on current state
        action, _, _ = dda_agent.select_action(state, deterministic=True)
        
        # Step DDA environment (it handles game steps internally)
        state, dda_reward, done, info = dda_env.step(action)
        
        # Get current game state
        if game_env and hasattr(game_env, 'player') and isinstance(game_env.player, dict):
            current_health = game_env.player.get('health', 100.0)
            current_zombies = len(game_env.zombies) if hasattr(game_env, 'zombies') else 0
            current_kills = game_env.zombies_killed if hasattr(game_env, 'zombies_killed') else info.get('zombies_killed', 0)
        else:
            current_health = info.get('player_health', 100.0)
            current_zombies = info.get('zombies_alive', 0)
            current_kills = info.get('zombies_killed', 0)
        
        # Track health history (sample every step, DDA env steps multiple game frames)
        health_history.append(current_health)
        step_count += 1
        
        # Check if DDA action was applied (check dda_action_steps in environment)
        # The DDA environment applies actions when dda_action_steps >= action_interval
        # We detect this by checking if enough steps have passed
        dda_env_internal = dda_env.dda_env if hasattr(dda_env, 'dda_env') else dda_env
        if hasattr(dda_env_internal, 'dda_action_steps'):
            dda_action_steps = dda_env_internal.dda_action_steps
            # When dda_action_steps resets to 0 or is small, a new action was just applied
            if dda_action_steps < 50 and last_dda_action_step >= 250:  # Action just applied
                # Save previous interval if not first
                if current_interval > 0:
                    interval_kills = current_kills - interval_start_kills
                    interval_damage_taken = info.get('total_damage_taken', 0.0) - interval_start_damage_taken
                    interval_damage_dealt = info.get('total_damage_dealt', 0.0) - interval_start_damage_dealt
                    
                    interval_metrics.append({
                        'game_id': game_id,
                        'interval_number': current_interval,
                        'action_name': ACTION_NAMES.get(last_action, "Unknown"),
                        'health_start': interval_start_health,
                        'health_end': current_health,
                        'kills_interval': max(0, interval_kills),
                        'damage_taken_interval': max(0.0, interval_damage_taken),
                        'damage_dealt_interval': max(0.0, interval_damage_dealt),
                        'zombies_alive_start': interval_start_zombies
                    })
                
                # Start new interval
                current_interval += 1
                last_action = action
                action_history.append(action)
                action_counts[action] += 1
                
                # Save interval start metrics
                interval_start_health = current_health
                interval_start_zombies = current_zombies
                interval_start_kills = current_kills
                # Get damage metrics from DDA environment if available
                dda_env_internal = dda_env.dda_env if hasattr(dda_env, 'dda_env') else dda_env
                if hasattr(dda_env_internal, 'total_damage_taken'):
                    interval_start_damage_taken = dda_env_internal.total_damage_taken
                else:
                    interval_start_damage_taken = 0.0
                if hasattr(dda_env_internal, 'total_damage_dealt'):
                    interval_start_damage_dealt = dda_env_internal.total_damage_dealt
                else:
                    interval_start_damage_dealt = 0.0
            
            last_dda_action_step = dda_action_steps
        else:
            # Fallback: approximate intervals every 300 steps
            if current_interval == 0 or step_count % 300 == 0:
                if current_interval > 0:
                    interval_kills = current_kills - interval_start_kills
                    dda_env_internal = dda_env.dda_env if hasattr(dda_env, 'dda_env') else dda_env
                    if hasattr(dda_env_internal, 'total_damage_taken'):
                        interval_damage_taken = dda_env_internal.total_damage_taken - interval_start_damage_taken
                    else:
                        interval_damage_taken = 0.0
                    if hasattr(dda_env_internal, 'total_damage_dealt'):
                        interval_damage_dealt = dda_env_internal.total_damage_dealt - interval_start_damage_dealt
                    else:
                        interval_damage_dealt = 0.0
                    
                    interval_metrics.append({
                        'game_id': game_id,
                        'interval_number': current_interval,
                        'action_name': ACTION_NAMES.get(last_action, "Unknown"),
                        'health_start': interval_start_health,
                        'health_end': current_health,
                        'kills_interval': max(0, interval_kills),
                        'damage_taken_interval': max(0.0, interval_damage_taken),
                        'damage_dealt_interval': max(0.0, interval_damage_dealt),
                        'zombies_alive_start': interval_start_zombies
                    })
                
                current_interval += 1
                last_action = action
                action_history.append(action)
                action_counts[action] += 1
                
                interval_start_health = current_health
                interval_start_zombies = current_zombies
                interval_start_kills = current_kills
                dda_env_internal = dda_env.dda_env if hasattr(dda_env, 'dda_env') else dda_env
                if hasattr(dda_env_internal, 'total_damage_taken'):
                    interval_start_damage_taken = dda_env_internal.total_damage_taken
                else:
                    interval_start_damage_taken = 0.0
                if hasattr(dda_env_internal, 'total_damage_dealt'):
                    interval_start_damage_dealt = dda_env_internal.total_damage_dealt
                else:
                    interval_start_damage_dealt = 0.0
        
        if done:
            break
    
    # Save last interval metrics
    if current_interval > 0:
        if game_env and hasattr(game_env, 'zombies_killed'):
            current_kills = game_env.zombies_killed
        else:
            current_kills = info.get('zombies_killed', 0)
        
        dda_env_internal = dda_env.dda_env if hasattr(dda_env, 'dda_env') else dda_env
        if hasattr(dda_env_internal, 'total_damage_taken'):
            current_damage_taken = dda_env_internal.total_damage_taken
        else:
            current_damage_taken = 0.0
        if hasattr(dda_env_internal, 'total_damage_dealt'):
            current_damage_dealt = dda_env_internal.total_damage_dealt
        else:
            current_damage_dealt = 0.0
        
        interval_kills = current_kills - interval_start_kills
        interval_damage_taken = current_damage_taken - interval_start_damage_taken
        interval_damage_dealt = current_damage_dealt - interval_start_damage_dealt
        
        interval_metrics.append({
            'game_id': game_id,
            'interval_number': current_interval,
            'action_name': ACTION_NAMES.get(last_action, "Unknown"),
            'health_start': interval_start_health,
            'health_end': health_history[-1] if health_history else 100.0,
            'kills_interval': max(0, interval_kills),
            'damage_taken_interval': max(0.0, interval_damage_taken),
            'damage_dealt_interval': max(0.0, interval_damage_dealt),
            'zombies_alive_start': interval_start_zombies
        })
    
    # Calculate per-game metrics
    if not health_history:
        health_history = [100.0]
    
    health_array = np.array(health_history)
    avg_health = float(np.mean(health_array))
    health_std = float(np.std(health_array))
    min_health = float(np.min(health_array))
    
    # Flow zone metrics (30-70% health)
    time_in_flow_zone = np.sum((health_array >= 30) & (health_array <= 70))
    time_in_flow_zone_percent = (time_in_flow_zone / len(health_array)) * 100.0
    
    # Danger zone metrics (<30% health)
    time_in_danger = np.sum(health_array < 30)
    time_in_danger_zone_percent = (time_in_danger / len(health_array)) * 100.0
    
    # Safety zone metrics (>70% health)
    time_in_safety = np.sum(health_array > 70)
    time_in_safety_zone_percent = (time_in_safety / len(health_array)) * 100.0
    
    # Near-death metrics (<20% health)
    # Count: number of times health dropped below 20%
    near_death_mask = health_array < 20
    near_death_count = 0
    was_below_20 = False
    for below_20 in near_death_mask:
        if below_20 and not was_below_20:
            near_death_count += 1
        was_below_20 = below_20
    
    # Time: percentage of time spent below 20%
    near_death_time = np.sum(near_death_mask)
    near_death_time_percent = (near_death_time / len(health_array)) * 100.0
    
    # Action distribution
    action_do_nothing = action_counts.get(0, 0)
    action_slightly_easier = action_counts.get(1, 0)
    action_much_easier = action_counts.get(2, 0)
    action_slightly_harder = action_counts.get(3, 0)
    action_much_harder = action_counts.get(4, 0)
    
    # Action oscillations
    action_oscillations = calculate_action_oscillations(action_history)
    
    # Get final metrics from game environment or info
    if game_env and hasattr(game_env, 'zombies_killed'):
        total_kills = game_env.zombies_killed
    else:
        total_kills = info.get('zombies_killed', 0)
    
    # Survival time is the number of steps (DDA env steps, which are ~300 game frames each)
    # Approximate by number of health samples (each sample represents some game frames)
    survival_time = len(health_history) * 300  # Approximate: each DDA step = ~300 game frames
    
    game_metrics = {
        'game_id': game_id,
        'survival_time_steps': survival_time,
        'total_kills': total_kills,
        'avg_health_percent': avg_health,
        'health_std': health_std,
        'min_health': min_health,
        'time_in_flow_zone_percent': time_in_flow_zone_percent,
        'time_in_danger_zone_percent': time_in_danger_zone_percent,
        'time_in_safety_zone_percent': time_in_safety_zone_percent,
        'near_death_count': int(near_death_count),
        'near_death_time_percent': near_death_time_percent,
        'action_do_nothing_count': action_do_nothing,
        'action_slightly_easier_count': action_slightly_easier,
        'action_much_easier_count': action_much_easier,
        'action_slightly_harder_count': action_slightly_harder,
        'action_much_harder_count': action_much_harder,
        'action_oscillations': action_oscillations
    }
    
    return game_metrics, interval_metrics


def evaluate_dda_agent(model_info: Dict, player_agent, num_games: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Evaluate a single DDA agent by running multiple games.
    
    Returns:
        game_results: List of per-game metrics
        interval_results: List of per-interval metrics
    """
    model_path = model_info['path']
    model_name = model_info['name']
    is_cnn = model_info['type'] == 'cnn'
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Type: {'CNN-based' if is_cnn else 'RL-based (Feature)'}")
    print(f"{'='*70}")
    
    # Create configs
    ppo_config = PPOConfig()
    ppo_config.headless = True  # Run headless for faster evaluation
    
    # Create game environment
    game_env = ZombieShooterEnv(ppo_config, headless=True)
    
    # Get base parameters
    base_params = get_base_params()
    
    # Create DDA environment (feature-based, needed for game mechanics)
    dda_env_feature = DDAEnvironment(
        game_env=game_env,
        base_params=base_params,
        action_interval=300,  # 5 seconds at 60 FPS
        fps=60
    )
    
    # Set player agent
    dda_env_feature.set_player_agent(player_agent)
    
    # Create DDA agent
    if is_cnn:
        dda_config = CNNDDAConfig()
        dda_agent = CNNDDAAgent(dda_config)
        
        # Wrap with CNN wrapper
        dda_env = CNNDDAEnvWrapper(dda_env_feature)
    else:
        dda_config = DDAConfig()
        dda_agent = DDAAgent(dda_config)
        dda_env = dda_env_feature
    
    # Load DDA model
    try:
        dda_agent.load(model_path)
        print(f"✓ DDA model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading DDA model: {e}")
        return [], []
    
    # Run games
    all_game_metrics = []
    all_interval_metrics = []
    
    game_range = tqdm(range(num_games), desc=f"Running games for {model_name}", leave=False) if HAS_TQDM else range(num_games)
    
    for game_id in game_range:
        try:
            game_metrics, interval_metrics = run_game_with_dda(
                dda_agent, dda_env, is_cnn, game_id + 1
            )
            
            # Add model name to metrics
            game_metrics['dda_agent_name'] = model_name
            for interval in interval_metrics:
                interval['dda_agent_name'] = model_name
            
            all_game_metrics.append(game_metrics)
            all_interval_metrics.extend(interval_metrics)
            
            # Print progress every 10 games if no tqdm
            if not HAS_TQDM and (game_id + 1) % 10 == 0:
                print(f"  Completed {game_id + 1}/{num_games} games")
        
        except Exception as e:
            print(f"  Error in game {game_id + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Clean up
    game_env.close()
    
    print(f"✓ Completed {len(all_game_metrics)} games for {model_name}")
    
    return all_game_metrics, all_interval_metrics


def save_results(game_results: List[Dict], interval_results: List[Dict], output_dir: str):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save game-level results
    game_file = os.path.join(output_dir, "dda_evaluation_agents.csv")
    if game_results:
        fieldnames = [
            'dda_agent_name', 'game_id', 'survival_time_steps', 'total_kills',
            'avg_health_percent', 'health_std', 'min_health',
            'time_in_flow_zone_percent', 'time_in_danger_zone_percent', 'time_in_safety_zone_percent',
            'near_death_count', 'near_death_time_percent',
            'action_do_nothing_count', 'action_slightly_easier_count', 'action_much_easier_count',
            'action_slightly_harder_count', 'action_much_harder_count', 'action_oscillations'
        ]
        
        with open(game_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_results)
        
        print(f"\n✓ Game-level results saved to: {game_file}")
    
    # Save interval-level results
    interval_file = os.path.join(output_dir, "dda_evaluation_intervals.csv")
    if interval_results:
        fieldnames = [
            'dda_agent_name', 'game_id', 'interval_number', 'action_name',
            'health_start', 'health_end', 'kills_interval',
            'damage_taken_interval', 'damage_dealt_interval', 'zombies_alive_start'
        ]
        
        with open(interval_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(interval_results)
        
        print(f"✓ Interval-level results saved to: {interval_file}")


def print_summary_statistics(game_results: List[Dict]):
    """Print summary statistics for each DDA agent."""
    if not game_results:
        return
    
    # Group by agent
    agents = {}
    for result in game_results:
        agent_name = result['dda_agent_name']
        if agent_name not in agents:
            agents[agent_name] = []
        agents[agent_name].append(result)
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for agent_name, results in agents.items():
        print(f"\n{agent_name} ({len(results)} games):")
        print("-" * 70)
        
        # Key metrics
        survival_times = [r['survival_time_steps'] for r in results]
        total_kills = [r['total_kills'] for r in results]
        avg_health = [r['avg_health_percent'] for r in results]
        flow_zone = [r['time_in_flow_zone_percent'] for r in results]
        danger_zone = [r['time_in_danger_zone_percent'] for r in results]
        
        print(f"Survival Time (steps):")
        print(f"  Mean: {np.mean(survival_times):.1f}, Std: {np.std(survival_times):.1f}")
        print(f"  Min: {np.min(survival_times):.0f}, Max: {np.max(survival_times):.0f}")
        
        print(f"Total Kills:")
        print(f"  Mean: {np.mean(total_kills):.1f}, Std: {np.std(total_kills):.1f}")
        print(f"  Min: {np.min(total_kills):.0f}, Max: {np.max(total_kills):.0f}")
        
        print(f"Average Health (%):")
        print(f"  Mean: {np.mean(avg_health):.1f}, Std: {np.std(avg_health):.1f}")
        print(f"  Min: {np.min(avg_health):.1f}, Max: {np.max(avg_health):.1f}")
        
        print(f"Time in Flow Zone (%):")
        print(f"  Mean: {np.mean(flow_zone):.1f}, Std: {np.std(flow_zone):.1f}")
        print(f"  Min: {np.min(flow_zone):.1f}, Max: {np.max(flow_zone):.1f}")
        
        print(f"Time in Danger Zone (%):")
        print(f"  Mean: {np.mean(danger_zone):.1f}, Std: {np.std(danger_zone):.1f}")
        print(f"  Min: {np.min(danger_zone):.1f}, Max: {np.max(danger_zone):.1f}")


def main():
    """Main evaluation function."""
    print("="*70)
    print("DDA AGENTS EVALUATION")
    print("="*70)
    
    # Configuration
    player_model_path = os.path.join(project_root, "checkpoints", "player_ppo", "best_model_marouane.pth")
    num_games = 100
    # Save results to local data/evaluation directory (relative to script location)
    output_dir = os.path.join(script_dir, "data", "evaluation")
    
    # Load player agent
    print(f"\nLoading player agent from: {player_model_path}")
    ppo_config = PPOConfig()
    player_agent = PPOAgent(ppo_config)
    
    try:
        player_agent.load(player_model_path)
        print("✓ Player agent loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Player model not found at {player_model_path}")
        print("Please ensure the player model exists.")
        return
    except Exception as e:
        print(f"✗ Error loading player agent: {e}")
        return
    
    # Find all DDA models
    print(f"\nSearching for DDA models...")
    dda_models = find_dda_models()
    
    if not dda_models:
        print("✗ No DDA models found!")
        print("  Searched in:")
        print(f"    - {os.path.join(project_root, 'checkpoints', 'dda')}")
        print(f"    - {os.path.join(project_root, 'checkpoints_cnn', 'dda_cnn')}")
        return
    
    print(f"✓ Found {len(dda_models)} DDA model(s):")
    for model in dda_models:
        print(f"    - {model['name']} ({model['type'].upper()})")
    
    # Evaluate each DDA agent
    all_game_results = []
    all_interval_results = []
    
    model_iter = tqdm(dda_models, desc="Evaluating DDA agents", unit="agent") if HAS_TQDM else dda_models
    
    for model_info in model_iter:
        game_results, interval_results = evaluate_dda_agent(
            model_info, player_agent, num_games=num_games
        )
        all_game_results.extend(game_results)
        all_interval_results.extend(interval_results)
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    save_results(all_game_results, all_interval_results, output_dir)
    
    # Print summary
    print_summary_statistics(all_game_results)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

