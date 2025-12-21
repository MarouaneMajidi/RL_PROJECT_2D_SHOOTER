"""
Metrics Collector for DDA Evaluation

Collects all metrics defined in DDA_EVALUATION_METRICS.md during gameplay.
"""

import json
import csv
import os
import math
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


class MetricsCollector:
    """Collects and stores metrics for DDA evaluation."""
    
    def __init__(self, player_id: str, match_number: int, condition: str, output_dir: str = "data/evaluation"):
        """
        Initialize metrics collector.
        
        Args:
            player_id: Unique identifier for the player
            match_number: Match number (1 or 2)
            condition: "with_dda" or "without_dda"
            output_dir: Directory to save collected data
        """
        self.player_id = player_id
        self.match_number = match_number
        self.condition = condition
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Time tracking
        self.start_time = None
        self.end_time = None
        self.match_duration = 0.0
        
        # Frame-by-frame data (collected every second)
        self.frame_data = []
        
        # Event tracking
        self.damage_taken_history = []
        self.damage_dealt_history = []
        self.kills_history = []
        self.shots_fired_history = []
        self.shots_hit_history = []
        self.pickups_collected_history = []
        self.health_history = []
        self.zombies_alive_history = []
        self.distance_moved = 0.0
        self.last_position = None
        
        # DDA-specific tracking (only for with_dda condition)
        self.dda_actions = []
        self.dda_action_times = []
        self.difficulty_params_history = []
        
        # Final metrics
        self.final_score = 0
        self.total_kills = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.total_pickups_collected = 0
        self.min_health = 100
        
    def start_match(self):
        """Start recording metrics."""
        self.start_time = datetime.now()
        
    def end_match(self):
        """End recording and calculate final metrics."""
        self.end_time = datetime.now()
        if self.start_time:
            self.match_duration = (self.end_time - self.start_time).total_seconds()
    
    def record_frame(self, game_state: Dict):
        """
        Record game state at current frame (called every second).
        
        Args:
            game_state: Dictionary containing:
                - player_health: Current health (0-100)
                - zombies_alive: Number of zombies currently alive
                - player_x: Player x position
                - player_y: Player y position
                - score: Current score
                - damage_taken: Damage taken this frame
                - damage_dealt: Damage dealt this frame
                - zombie_killed: Whether a zombie was killed this frame
                - shot_fired: Whether a shot was fired this frame
                - shot_hit: Whether a shot hit this frame
                - pickup_collected: Whether a pickup was collected this frame
                - dda_action: DDA action taken (if with_dda)
                - difficulty_params: Current difficulty parameters (if with_dda)
        """
        current_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        
        # Record frame data
        frame_entry = {
            'time': current_time,
            'player_health': game_state.get('player_health', 100),
            'zombies_alive': game_state.get('zombies_alive', 0),
            'player_x': game_state.get('player_x', 0),
            'player_y': game_state.get('player_y', 0),
            'score': game_state.get('score', 0),
        }
        
        # Track health
        health = game_state.get('player_health', 100)
        self.health_history.append(health)
        if health < self.min_health:
            self.min_health = health
        
        # Track zombies alive
        self.zombies_alive_history.append(game_state.get('zombies_alive', 0))
        
        # Track distance moved
        player_x = game_state.get('player_x', 0)
        player_y = game_state.get('player_y', 0)
        if self.last_position:
            dx = player_x - self.last_position[0]
            dy = player_y - self.last_position[1]
            self.distance_moved += math.sqrt(dx*dx + dy*dy)
        self.last_position = (player_x, player_y)
        
        # Track events
        if game_state.get('damage_taken', 0) > 0:
            self.damage_taken_history.append(game_state['damage_taken'])
            self.total_damage_taken += game_state['damage_taken']
        
        if game_state.get('damage_dealt', 0) > 0:
            self.damage_dealt_history.append(game_state['damage_dealt'])
            self.total_damage_dealt += game_state['damage_dealt']
        
        if game_state.get('zombie_killed', False):
            self.kills_history.append(1)
            self.total_kills += 1
        
        if game_state.get('shot_fired', False):
            self.shots_fired_history.append(1)
        
        if game_state.get('shot_hit', False):
            self.shots_hit_history.append(1)
        
        if game_state.get('pickup_collected', False):
            self.pickups_collected_history.append(1)
            self.total_pickups_collected += 1
        
        # Update final score
        self.final_score = game_state.get('score', 0)
        
        # DDA-specific tracking
        if self.condition == "with_dda":
            dda_action = game_state.get('dda_action')
            if dda_action is not None:
                self.dda_actions.append(dda_action)
                self.dda_action_times.append(current_time)
            
            difficulty_params = game_state.get('difficulty_params')
            if difficulty_params:
                self.difficulty_params_history.append({
                    'time': current_time,
                    **difficulty_params
                })
        
        self.frame_data.append(frame_entry)
    
    def calculate_metrics(self) -> Dict:
        """Calculate all metrics from collected data."""
        if not self.frame_data:
            return {}
        
        metrics = {}
        
        # 1. SURVIVAL METRICS
        metrics['survival_time'] = self.match_duration
        metrics['final_score'] = self.final_score
        
        # 2. HEALTH METRICS (Flow Zone)
        if self.health_history:
            metrics['average_health'] = np.mean(self.health_history)
            metrics['health_variance'] = np.std(self.health_history)
            metrics['min_health'] = self.min_health
            
            # Time in flow zone (30-70%)
            flow_zone_time = sum(1 for h in self.health_history if 30 <= h <= 70)
            metrics['time_in_flow_zone_percent'] = (flow_zone_time / len(self.health_history)) * 100
            
            # Time in danger (< 30%)
            danger_time = sum(1 for h in self.health_history if h < 30)
            metrics['time_in_danger_percent'] = (danger_time / len(self.health_history)) * 100
            
            # Time in safety (> 70%)
            safety_time = sum(1 for h in self.health_history if h > 70)
            metrics['time_in_safety_percent'] = (safety_time / len(self.health_history)) * 100
        
        # 3. PERFORMANCE METRICS
        metrics['total_kills'] = self.total_kills
        if self.match_duration > 0:
            metrics['kill_rate_per_minute'] = (self.total_kills / self.match_duration) * 60
        else:
            metrics['kill_rate_per_minute'] = 0
        
        # Shooting accuracy
        total_shots = sum(self.shots_fired_history)
        total_hits = sum(self.shots_hit_history)
        metrics['shooting_accuracy'] = (total_hits / total_shots * 100) if total_shots > 0 else 0
        
        metrics['total_damage_dealt'] = self.total_damage_dealt
        metrics['total_damage_taken'] = self.total_damage_taken
        
        # 4. DIFFICULTY METRICS
        if self.zombies_alive_history:
            metrics['average_zombies_alive'] = np.mean(self.zombies_alive_history)
            metrics['peak_zombies_alive'] = max(self.zombies_alive_history)
        
        # 5. ENGAGEMENT METRICS
        metrics['total_pickups_collected'] = self.total_pickups_collected
        metrics['total_distance_moved'] = self.distance_moved
        if self.match_duration > 0:
            metrics['average_movement_speed'] = self.distance_moved / self.match_duration
        else:
            metrics['average_movement_speed'] = 0
        
        # 6. DDA BEHAVIOR METRICS (only for with_dda)
        if self.condition == "with_dda":
            metrics['total_dda_adjustments'] = len(self.dda_actions)
            if self.match_duration > 0:
                metrics['dda_adjustment_frequency'] = (len(self.dda_actions) / self.match_duration) * 60  # per minute
            
            # Action distribution
            action_counts = {}
            for action in self.dda_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            total_actions = len(self.dda_actions)
            if total_actions > 0:
                metrics['dda_action_distribution'] = {
                    action: (count / total_actions) * 100 
                    for action, count in action_counts.items()
                }
            
            # Oscillations (changes in direction)
            oscillations = 0
            for i in range(1, len(self.dda_actions)):
                prev = self.dda_actions[i-1]
                curr = self.dda_actions[i]
                # Check if direction changed (easier -> harder or harder -> easier)
                if ("Easier" in prev and "Harder" in curr) or ("Harder" in prev and "Easier" in curr):
                    oscillations += 1
            metrics['dda_oscillations'] = oscillations
            if total_actions > 0:
                metrics['dda_oscillation_percent'] = (oscillations / total_actions) * 100
        
        return metrics
    
    def save_data(self):
        """Save collected data to files."""
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        # Save frame-by-frame data
        frame_file = os.path.join(
            self.output_dir, 
            f"{self.player_id}_match{self.match_number}_{self.condition}_frames.json"
        )
        with open(frame_file, 'w') as f:
            json.dump(self.frame_data, f, indent=2)
        
        # Save calculated metrics
        metrics_file = os.path.join(
            self.output_dir,
            f"{self.player_id}_match{self.match_number}_{self.condition}_metrics.json"
        )
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save to CSV for easy analysis
        csv_file = os.path.join(
            self.output_dir,
            f"{self.player_id}_match{self.match_number}_{self.condition}_metrics.csv"
        )
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}_{sub_key}", sub_value])
                else:
                    writer.writerow([key, value])
        
        return metrics_file, csv_file
    
    def save_to_consolidated(self):
        """Save metrics to consolidated files (all players in same files)."""
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        # Prepare entry with player info
        entry = {
            'player_id': self.player_id,
            'match_number': self.match_number,
            'condition': self.condition,
            'timestamp': datetime.now().isoformat(),
            'match_duration': self.match_duration,
            **metrics
        }
        
        # Append to consolidated metrics file
        consolidated_file = os.path.join(self.output_dir, "all_metrics.json")
        if os.path.exists(consolidated_file):
            with open(consolidated_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        all_metrics.append(entry)
        
        with open(consolidated_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Also save frame data to consolidated file
        frame_entry = {
            'player_id': self.player_id,
            'match_number': self.match_number,
            'condition': self.condition,
            'timestamp': datetime.now().isoformat(),
            'frames': self.frame_data
        }
        
        consolidated_frames_file = os.path.join(self.output_dir, "all_frames.json")
        if os.path.exists(consolidated_frames_file):
            with open(consolidated_frames_file, 'r') as f:
                all_frames = json.load(f)
        else:
            all_frames = []
        
        all_frames.append(frame_entry)
        
        with open(consolidated_frames_file, 'w') as f:
            json.dump(all_frames, f, indent=2)
        
        return consolidated_file, consolidated_frames_file

