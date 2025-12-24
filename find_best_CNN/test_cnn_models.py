"""
Test Script for Multiple CNN Models

Tests all CNN models in checkpoints_cnn/player_cnn and collects performance metrics.
Saves results to CSV file.
"""

import os
import sys
import csv

# Try to import tqdm for progress bars, but make it optional
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents_cnn.player_cnn.agent import CNNPlayerAgent
from agents_cnn.player_cnn.config import CNNPlayerConfig
from agents_cnn.player_cnn.env_wrapper import CNNEnvWrapper
from agents.ppo_agent.config import PPOConfig


def get_model_files():
    """Get all CNN model files from checkpoints directory."""
    checkpoint_dir = os.path.join(project_root, "checkpoints_cnn", "player_cnn")
    model_files = []
    
    # Define the 5 models we want to test
    model_names = [
        "best_model.pth",
        "best_model_stay_up_1.pth",
        "best_model_stay_up_2.pth",
        "best_model_stay_up_4.pth",
        "checkpoint_550912_113killed_wave12_3.pth"
    ]
    
    for model_name in model_names:
        model_path = os.path.join(checkpoint_dir, model_name)
        if os.path.exists(model_path):
            model_files.append({
                'path': model_path,
                'name': model_name.replace('.pth', '')  # Remove .pth extension
            })
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    return model_files


def test_model(model_path: str, model_name: str, num_episodes: int = 100):
    """
    Test a single CNN model and collect metrics.
    
    Args:
        model_path: Path to the model checkpoint
        model_name: Name of the model (for CSV)
        num_episodes: Number of episodes to test
        
    Returns:
        List of dictionaries with metrics for each episode
    """
    results = []
    
    try:
        # Create config
        config = CNNPlayerConfig()
        config.headless = True  # Run headless for faster testing
        
        # Create PPO config for environment
        ppo_config = PPOConfig()
        
        # Create CNN environment wrapper
        env = CNNEnvWrapper(ppo_config, headless=True)
        
        # Create CNN agent
        agent = CNNPlayerAgent(config)
        
        # Load model
        agent.load(model_path)
        
        # Test for num_episodes with progress bar
        episode_range = tqdm(range(num_episodes), desc=f"Testing {model_name}", leave=False) if HAS_TQDM else range(num_episodes)
        
        for episode in episode_range:
            state = env.reset()
            done = False
            survival_time = 0
            
            while not done:
                # Select action (deterministic for consistent testing)
                action, _, _ = agent.select_action(state, deterministic=True)
                
                # Take step
                state, reward, done, info = env.step(action)
                
                survival_time += 1
            
            # Collect metrics from info dict
            zombies_killed = info.get('zombies_killed', 0)
            wave_achieved = info.get('wave', 1)
            
            results.append({
                'model_name': model_name,
                'episode': episode + 1,
                'survival_time': survival_time,
                'zombies_killed': zombies_killed,
                'wave_achieved': wave_achieved
            })
            
            # Print progress every 10 episodes if no tqdm
            if not HAS_TQDM and (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{num_episodes} completed")
        
        env.close()
        
    except Exception as e:
        import traceback
        print(f"\nError testing model {model_name}: {e}")
        traceback.print_exc()
        # Add error entries for remaining episodes
        for episode in range(len(results), num_episodes):
            results.append({
                'model_name': model_name,
                'episode': episode + 1,
                'survival_time': -1,  # Error marker
                'zombies_killed': -1,
                'wave_achieved': -1
            })
    
    return results


def save_results_to_csv(results: list, output_file: str):
    """Save test results to CSV file."""
    fieldnames = ['model_name', 'episode', 'survival_time', 'zombies_killed', 'wave_achieved']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")


def print_summary(results: list, model_name: str):
    """Print summary statistics for a model."""
    model_results = [r for r in results if r['model_name'] == model_name and r['survival_time'] >= 0]
    
    if not model_results:
        print(f"  No valid results for {model_name}")
        return
    
    survival_times = [r['survival_time'] for r in model_results]
    zombies_killed = [r['zombies_killed'] for r in model_results]
    waves = [r['wave_achieved'] for r in model_results]
    
    print(f"\n{model_name} Summary:")
    print(f"  Episodes completed: {len(model_results)}")
    print(f"  Avg Survival Time: {sum(survival_times) / len(survival_times):.1f} steps")
    print(f"  Avg Zombies Killed: {sum(zombies_killed) / len(zombies_killed):.1f}")
    print(f"  Avg Wave Achieved: {sum(waves) / len(waves):.2f}")
    print(f"  Max Survival Time: {max(survival_times)} steps")
    print(f"  Max Zombies Killed: {max(zombies_killed)}")
    print(f"  Max Wave: {max(waves)}")


def main():
    """Main function."""
    print("=" * 70)
    print("CNN Model Testing Script")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(project_root, "find_best_CNN")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "test_results.csv")
    
    # Get all model files
    model_files = get_model_files()
    
    if not model_files:
        print("Error: No model files found!")
        return
    
    print(f"\nFound {len(model_files)} models to test:")
    for m in model_files:
        print(f"  - {m['name']}")
    
    print(f"\nTesting each model for 100 episodes...")
    print(f"Results will be saved to: {output_file}")
    print("=" * 70)
    
    all_results = []
    
    # Test each model
    model_iter = tqdm(model_files, desc="Testing Models", unit="model") if HAS_TQDM else model_files
    for model_info in model_iter:
        model_path = model_info['path']
        model_name = model_info['name']
        
        print(f"\n{'=' * 70}")
        print(f"Testing Model: {model_name}")
        print(f"{'=' * 70}")
        
        # Test with progress bar
        print(f"Running 100 episodes...")
        results = test_model(model_path, model_name, num_episodes=100)
        all_results.extend(results)
        
        # Print summary for this model
        print_summary(all_results, model_name)
    
    # Save all results to CSV
    save_results_to_csv(all_results, output_file)
    
    # Print overall summary
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    
    for model_info in model_files:
        model_name = model_info['name']
        print_summary(all_results, model_name)
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
