"""
Difficulty Manager

Manages game difficulty parameters grouped into categories.
Applies smooth, bounded adjustments based on DDA agent actions.
"""

from typing import Dict, Tuple
import numpy as np


class DifficultyManager:
    """
    Manages difficulty parameters grouped into categories.
    
    Groups:
    1. Enemy Pressure: ZOMBIE_SPEED, SPAWN_RATE, ZOMBIE_ATTACK_COOLDOWN
    2. Enemy Durability: ZOMBIE_HEALTH
    3. Player Power: PLAYER_SPEED, WEAPON_DAMAGE
    4. Resource Generosity: HEALTH_PICKUP_DROP_PROBABILITY, MACHINEGUN_PICKUP_DROP_PROBABILITY
    5. Combat Tempo: WEAPON_COOLDOWNS
    """
    
    def __init__(self, base_params: Dict):
        """
        Initialize difficulty manager with base parameters.
        
        Args:
            base_params: Dictionary with base game parameters:
                - ZOMBIE_SPEED
                - SPAWN_RATE
                - ZOMBIE_ATTACK_COOLDOWN
                - ZOMBIE_NORMAL_HEALTH
                - ZOMBIE_STRONG_HEALTH
                - PLAYER_SPEED
                - PISTOL_DAMAGE
                - MACHINEGUN_DAMAGE
                - HEALTH_PICKUP_DROP_PROBABILITY
                - MACHINEGUN_PICKUP_DROP_PROBABILITY
                - PISTOL_COOLDOWN
                - MACHINEGUN_COOLDOWN
        """
        self.base_params = base_params.copy()
        self.current_params = base_params.copy()
        
        # Define parameter bounds (min, max) for each parameter
        self.bounds = {
            'ZOMBIE_SPEED': (1.0, 5.0),
            'SPAWN_RATE': (10, 120),  # Lower = faster spawn
            'ZOMBIE_ATTACK_COOLDOWN': (5, 15),
            'ZOMBIE_NORMAL_HEALTH': (20, 50),
            'ZOMBIE_STRONG_HEALTH': (40, 80),
            'PLAYER_SPEED': (3.0, 8.0),
            'PISTOL_DAMAGE': (15, 30),
            'MACHINEGUN_DAMAGE': (5, 20),
            'HEALTH_PICKUP_DROP_PROBABILITY': (0.1, 0.6),
            'MACHINEGUN_PICKUP_DROP_PROBABILITY': (0.05, 0.3),
            'PISTOL_COOLDOWN': (15, 30),
            'MACHINEGUN_COOLDOWN': (3, 10),
        }
        
        # Adjustment amounts per action (as multipliers or deltas)
        # Action 0: Do nothing (0.0)
        # Action 1: Slightly easier (-0.05)
        # Action 2: Much easier (-0.15)
        # Action 3: Slightly harder (+0.05)
        # Action 4: Much harder (+0.15)
        self.adjustment_amounts = {
            0: 0.0,
            1: -0.05,  # Slightly easier
            2: -0.15,  # Much easier
            3: +0.05,  # Slightly harder
            4: +0.15,  # Much harder
        }
        
        # Smoothing factor to prevent oscillations
        self.smoothing_factor = 0.3  # Only apply 30% of change per step
    
    def reset(self):
        """Reset all parameters to base values."""
        self.current_params = self.base_params.copy()
    
    def apply_action(self, action: int) -> Dict:
        """
        Apply DDA action to adjust difficulty parameters.
        
        Args:
            action: Discrete action (0-4):
                0: Do nothing
                1: Slightly easier
                2: Much easier
                3: Slightly harder
                4: Much harder
        
        Returns:
            Dictionary with updated parameters
        """
        if action not in self.adjustment_amounts:
            action = 0  # Default to do nothing
        
        adjustment = self.adjustment_amounts[action]
        
        # Apply adjustments to each parameter group
        # Use smoothing to prevent oscillations
        smoothed_adjustment = adjustment * self.smoothing_factor
        
        # 1. Enemy Pressure Group
        self._adjust_parameter('ZOMBIE_SPEED', smoothed_adjustment, is_multiplier=True)
        self._adjust_parameter('SPAWN_RATE', -smoothed_adjustment, is_multiplier=True)  # Negative: lower spawn rate = easier
        self._adjust_parameter('ZOMBIE_ATTACK_COOLDOWN', smoothed_adjustment, is_multiplier=True)
        
        # 2. Enemy Durability Group
        self._adjust_parameter('ZOMBIE_NORMAL_HEALTH', smoothed_adjustment, is_multiplier=True)
        self._adjust_parameter('ZOMBIE_STRONG_HEALTH', smoothed_adjustment, is_multiplier=True)
        
        # 3. Player Power Group
        self._adjust_parameter('PLAYER_SPEED', -smoothed_adjustment, is_multiplier=True)  # Negative: easier = faster player
        self._adjust_parameter('PISTOL_DAMAGE', -smoothed_adjustment, is_multiplier=True)  # Negative: easier = more damage
        self._adjust_parameter('MACHINEGUN_DAMAGE', -smoothed_adjustment, is_multiplier=True)
        
        # 4. Resource Generosity Group
        self._adjust_parameter('HEALTH_PICKUP_DROP_PROBABILITY', -smoothed_adjustment, is_multiplier=True)  # Negative: easier = more pickups
        self._adjust_parameter('MACHINEGUN_PICKUP_DROP_PROBABILITY', -smoothed_adjustment, is_multiplier=True)
        
        # 5. Combat Tempo Group
        self._adjust_parameter('PISTOL_COOLDOWN', smoothed_adjustment, is_multiplier=True)  # Positive: easier = faster cooldown (negative adjustment)
        self._adjust_parameter('MACHINEGUN_COOLDOWN', smoothed_adjustment, is_multiplier=True)
        
        return self.current_params.copy()
    
    def _adjust_parameter(self, param_name: str, adjustment: float, is_multiplier: bool = True):
        """
        Adjust a single parameter within bounds.
        
        Args:
            param_name: Name of parameter to adjust
            adjustment: Adjustment amount (multiplier if is_multiplier=True, else delta)
            is_multiplier: If True, adjustment is a multiplier (1.0 + adjustment), else it's a delta
        """
        if param_name not in self.current_params:
            return
        
        base_value = self.base_params[param_name]
        current_value = self.current_params[param_name]
        min_val, max_val = self.bounds[param_name]
        
        if is_multiplier:
            # Apply as multiplier: new = current * (1.0 + adjustment)
            new_value = current_value * (1.0 + adjustment)
        else:
            # Apply as delta: new = current + adjustment
            new_value = current_value + adjustment
        
        # Clamp to bounds
        new_value = np.clip(new_value, min_val, max_val)
        
        # Also ensure we don't deviate too far from base (optional safety)
        # Allow up to 50% deviation from base
        base_min = base_value * 0.5
        base_max = base_value * 1.5
        new_value = np.clip(new_value, base_min, base_max)
        
        # Final clamp to absolute bounds
        new_value = np.clip(new_value, min_val, max_val)
        
        self.current_params[param_name] = float(new_value)
    
    def get_current_params(self) -> Dict:
        """Get current difficulty parameters."""
        return self.current_params.copy()
    
    def get_parameter_groups(self) -> Dict[str, Dict]:
        """
        Get parameters organized by groups.
        
        Returns:
            Dictionary with group names as keys and parameter dicts as values
        """
        return {
            'enemy_pressure': {
                'ZOMBIE_SPEED': self.current_params['ZOMBIE_SPEED'],
                'SPAWN_RATE': self.current_params['SPAWN_RATE'],
                'ZOMBIE_ATTACK_COOLDOWN': self.current_params['ZOMBIE_ATTACK_COOLDOWN'],
            },
            'enemy_durability': {
                'ZOMBIE_NORMAL_HEALTH': self.current_params['ZOMBIE_NORMAL_HEALTH'],
                'ZOMBIE_STRONG_HEALTH': self.current_params['ZOMBIE_STRONG_HEALTH'],
            },
            'player_power': {
                'PLAYER_SPEED': self.current_params['PLAYER_SPEED'],
                'PISTOL_DAMAGE': self.current_params['PISTOL_DAMAGE'],
                'MACHINEGUN_DAMAGE': self.current_params['MACHINEGUN_DAMAGE'],
            },
            'resource_generosity': {
                'HEALTH_PICKUP_DROP_PROBABILITY': self.current_params['HEALTH_PICKUP_DROP_PROBABILITY'],
                'MACHINEGUN_PICKUP_DROP_PROBABILITY': self.current_params['MACHINEGUN_PICKUP_DROP_PROBABILITY'],
            },
            'combat_tempo': {
                'PISTOL_COOLDOWN': self.current_params['PISTOL_COOLDOWN'],
                'MACHINEGUN_COOLDOWN': self.current_params['MACHINEGUN_COOLDOWN'],
            },
        }

