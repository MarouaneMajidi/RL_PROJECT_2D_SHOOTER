"""
DDA State Extractor

Extracts normalized state features from game metrics using sliding windows.
State represents current game situation, NOT full history.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class DDAStateExtractor:
    """
    Extracts state features for DDA agent from game metrics.
    
    Uses sliding windows to compute short-term aggregated metrics.
    All features are normalized to [0, 1].
    """
    
    def __init__(self, window_size: int = 600, fps: int = 60):
        """
        Initialize state extractor.
        
        Args:
            window_size: Size of sliding window in frames (default 600 = 10 seconds at 60 FPS)
            fps: Frames per second (for time-based calculations)
        """
        self.window_size = window_size
        self.fps = fps
        self.window_seconds = window_size / fps  # 10 seconds default
        
        # Sliding window buffers for metrics
        self.damage_taken_history = deque(maxlen=window_size)
        self.damage_dealt_history = deque(maxlen=window_size)
        self.kills_history = deque(maxlen=window_size)
        self.shots_fired_history = deque(maxlen=window_size)
        self.shots_hit_history = deque(maxlen=window_size)
        self.pickups_collected_history = deque(maxlen=window_size)
        
        # Current state tracking
        self.last_player_health = 100.0
        self.last_hit_time = 0  # Frame count since last player hit
        self.current_frame = 0
        
        # State dimension: 9 features as specified
        self.state_dim = 9
    
    def reset(self):
        """Reset all tracking buffers."""
        self.damage_taken_history.clear()
        self.damage_dealt_history.clear()
        self.kills_history.clear()
        self.shots_fired_history.clear()
        self.shots_hit_history.clear()
        self.pickups_collected_history.clear()
        
        self.last_player_health = 100.0
        self.last_hit_time = 0
        self.current_frame = 0
    
    def update(self, game_info: Dict):
        """
        Update metrics from game step.
        
        Args:
            game_info: Dictionary with game state information:
                - player_health: Current player health (0-100)
                - damage_taken: Damage taken this frame (0 if none)
                - damage_dealt: Damage dealt this frame (0 if none)
                - zombie_killed: True if zombie was killed this frame
                - shot_fired: True if shot was fired this frame
                - shot_hit: True if shot hit a zombie this frame
                - pickup_collected: True if pickup was collected this frame
        """
        self.current_frame += 1
        
        # Track damage taken
        if game_info.get('damage_taken', 0) > 0:
            self.damage_taken_history.append(game_info['damage_taken'])
            self.last_hit_time = 0
        else:
            self.damage_taken_history.append(0)
            self.last_hit_time += 1
        
        # Track damage dealt
        self.damage_dealt_history.append(game_info.get('damage_dealt', 0))
        
        # Track kills
        self.kills_history.append(1 if game_info.get('zombie_killed', False) else 0)
        
        # Track shooting accuracy
        if game_info.get('shot_fired', False):
            self.shots_fired_history.append(1)
            self.shots_hit_history.append(1 if game_info.get('shot_hit', False) else 0)
        else:
            self.shots_fired_history.append(0)
            self.shots_hit_history.append(0)
        
        # Track pickups
        self.pickups_collected_history.append(1 if game_info.get('pickup_collected', False) else 0)
        
        # Update last health
        self.last_player_health = game_info.get('player_health', 100.0)
    
    def extract_state(self, game_state: Dict) -> np.ndarray:
        """
        Extract normalized state vector from current game state.
        
        State features (all normalized to [0, 1]):
        0. player_health_ratio
        1. average_damage_taken_last_10s
        2. average_damage_dealt_last_10s
        3. kill_rate_last_10s
        4. shooting_accuracy_last_10s
        5. number_of_zombies_alive (normalized)
        6. time_since_last_player_hit (normalized)
        7. ammo_ratio
        8. pickups_collected_last_10s
        
        Args:
            game_state: Dictionary with current game state:
                - player_health: Current player health (0-100)
                - zombies_alive: Number of zombies currently alive
                - max_zombies: Maximum expected zombies (for normalization)
                - machinegun_ammo: Current machinegun ammo
                - max_ammo: Maximum ammo (for normalization)
        
        Returns:
            Normalized state vector of shape (9,)
        """
        # 0. Player health ratio
        player_health_ratio = game_state.get('player_health', 100.0) / 100.0
        
        # 1. Average damage taken in last window (normalized per second)
        if len(self.damage_taken_history) > 0:
            total_damage = sum(self.damage_taken_history)
            avg_damage_per_second = (total_damage / self.window_seconds) if self.window_seconds > 0 else 0.0
            # Normalize: assume max 50 damage per second is extreme
            average_damage_taken = min(avg_damage_per_second / 50.0, 1.0)
        else:
            average_damage_taken = 0.0
        
        # 2. Average damage dealt in last window (normalized per second)
        if len(self.damage_dealt_history) > 0:
            total_damage = sum(self.damage_dealt_history)
            avg_damage_per_second = (total_damage / self.window_seconds) if self.window_seconds > 0 else 0.0
            # Normalize: assume max 100 damage per second is high
            average_damage_dealt = min(avg_damage_per_second / 100.0, 1.0)
        else:
            average_damage_dealt = 0.0
        
        # 3. Kill rate in last window (kills per second)
        if len(self.kills_history) > 0:
            total_kills = sum(self.kills_history)
            kill_rate = (total_kills / self.window_seconds) if self.window_seconds > 0 else 0.0
            # Normalize: assume max 2 kills per second is high
            kill_rate_normalized = min(kill_rate / 2.0, 1.0)
        else:
            kill_rate_normalized = 0.0
        
        # 4. Shooting accuracy in last window
        if len(self.shots_fired_history) > 0:
            total_shots = sum(self.shots_fired_history)
            if total_shots > 0:
                total_hits = sum(self.shots_hit_history)
                shooting_accuracy = total_hits / total_shots
            else:
                shooting_accuracy = 0.0
        else:
            shooting_accuracy = 0.0
        
        # 5. Number of zombies alive (normalized)
        zombies_alive = game_state.get('zombies_alive', 0)
        max_zombies = game_state.get('max_zombies', 20)  # Default max
        number_of_zombies_alive = min(zombies_alive / max_zombies, 1.0)
        
        # 6. Time since last player hit (normalized)
        # Normalize: assume 5 seconds (300 frames) without hit is "safe"
        time_since_last_hit = min(self.last_hit_time / (5.0 * self.fps), 1.0)
        
        # 7. Ammo ratio
        machinegun_ammo = game_state.get('machinegun_ammo', 0)
        max_ammo = game_state.get('max_ammo', 100)  # Default max
        ammo_ratio = min(machinegun_ammo / max_ammo, 1.0) if max_ammo > 0 else 0.0
        
        # 8. Pickups collected in last window (normalized per second)
        if len(self.pickups_collected_history) > 0:
            total_pickups = sum(self.pickups_collected_history)
            pickup_rate = (total_pickups / self.window_seconds) if self.window_seconds > 0 else 0.0
            # Normalize: assume max 0.5 pickups per second is high
            pickups_collected = min(pickup_rate / 0.5, 1.0)
        else:
            pickups_collected = 0.0
        
        # Build state vector
        state = np.array([
            player_health_ratio,
            average_damage_taken,
            average_damage_dealt,
            kill_rate_normalized,
            shooting_accuracy,
            number_of_zombies_alive,
            time_since_last_hit,
            ammo_ratio,
            pickups_collected
        ], dtype=np.float32)
        
        # Ensure all values are in [0, 1]
        state = np.clip(state, 0.0, 1.0)
        
        return state

