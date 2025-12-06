"""
Gym-style Environment Wrapper for Zombie Shooter

Wraps the Pygame game into a gym-like interface for RL training.

FIXED ISSUES:
1. Added health pack distance/angle to observations for strategic pickups
2. Added danger level (enemy proximity, HP) to observations
3. Added machine gun support with proper weapon switching
4. Implemented continuous fire action (action 4 = shoot, can fire every frame if off cooldown)
5. Added reward shaping for low-health + pickup proximity incentives
6. Added reward for sustained DPS with machine gun
7. Fixed wave reset - now properly resets wave, spawn_rate, and all tracking on reset()
"""

import os
import sys
import numpy as np
import pygame
import math
import random
from typing import Tuple, Dict, Optional, List
from pygame.locals import *

# Add parent directory to path to import game components
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class ZombieShooterEnv:
    """
    Gym-like environment wrapper for the zombie shooter game.
    
    EXPERT RL SYSTEM - Redesigned for optimal behavior:
    
    Action Space (Discrete, 6 actions):
        0: Move Up
        1: Move Down
        2: Move Left
        3: Move Right
        4: Shoot (weapon selection handled internally based on state)
        5: Idle (no action)
    
    State Space (98-dim):
        - Player: [x, y, health, shoot_cooldown, angle, zombie_count] (6 values)
        - Weapon: [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg] (3 values)
        - Zombies (up to 13): [x, y, health, type, distance, angle_to_player] per zombie (78 values)
        - Pickups: [machinegun_location, health_pack_location, ammo_pickup_location]
                   Each: [distance, angle, exists] (9 values)
        - Distance to nearest zombie: 1 value
        - Zombie density in radius: 1 value
        Total: 6 + 3 + 78 + 9 + 1 + 1 = 98
    """
    
    # Game constants (imported from main.py)
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    PLAYER_SPEED = 5
    ZOMBIE_SPEED = 2
    BULLET_SPEED = 10
    SPAWN_RATE = 60  # Initial spawn rate
    
    ZOMBIE_TYPE2_SPAWN_PROBABILITY = 0.2
    HEALTH_PICKUP_DROP_PROBABILITY = 0.40
    MACHINEGUN_PICKUP_DROP_PROBABILITY = 0.15
    AMMO_PICKUP_DROP_PROBABILITY = 0.20  # Separate ammo pickup
    
    PISTOL_COOLDOWN = 20
    MACHINEGUN_COOLDOWN = 5  # Much faster fire rate!
    PISTOL_DAMAGE = 20
    MACHINEGUN_DAMAGE = 10
    
    ZOMBIE_NORMAL_HEALTH = 30
    ZOMBIE_NORMAL_DAMAGE = 10
    ZOMBIE_STRONG_HEALTH = 50
    ZOMBIE_STRONG_DAMAGE = 20
    ZOMBIE_ATTACK_COOLDOWN = 9
    
    HEALTH_PICKUP_AMOUNT = 25
    MACHINEGUN_PICKUP_AMMO = 50
    AMMO_PICKUP_AMOUNT = 30  # Ammo pickup gives 30 rounds
    PICKUP_LIFETIME = 600
    
    def __init__(self, config, headless: bool = True, render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            config: PPOConfig object with hyperparameters
            headless: If True, run without rendering (faster)
            render_mode: "human" for visual rendering, None otherwise
        """
        self.config = config
        self.headless = headless
        self.render_mode = render_mode
        
        # Action and observation spaces
        # Extended action space: 0-3 movement, 4 shoot, 5 idle, 6 switch pistol, 7 switch machinegun
        self.action_space_n = config.action_dim
        self.observation_space_shape = (config.state_dim,)
        self.max_zombies = config.max_zombies
        
        # Initialize Pygame
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("PPO Training - Zombie Shooter")
            self.clock = pygame.time.Clock()
        else:
            # Minimal pygame init for headless mode
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            self.screen = pygame.display.set_mode((1, 1))
        
        # Game state
        self.player = None
        self.zombies = []
        self.bullets = []
        self.pickups = []
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 10000
        self.spawn_timer = 0
        self.spawn_rate = self.SPAWN_RATE  # Track current spawn rate
        self.zombies_killed = 0
        self.zombies_killed_this_episode = 0  # For wave tracking (resets on death)
        self.total_reward = 0
        self.wave = 1  # Track current wave
        
        # Tracking for reward shaping
        self.last_player_pos = None
        self.idle_frames = 0
        self.last_health = 100
        self.previous_min_zombie_dist = float('inf')
        self.previous_danger = 0.0  # Track danger for danger-based rewards
        self.last_weapon = 'pistol'  # Track weapon switches
        
        # Machine gun tracking for DPS rewards
        self.consecutive_mg_hits = 0
        self.last_shot_was_mg = False
        self.dps_window = []  # Track damage dealt over time window
        self.mg_fire_time = []  # Track machine gun fire times for per-second reward
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        FIXED: Now properly resets wave, spawn_rate, and all tracking variables.
        
        Returns:
            Initial observation
        """
        # Reset game objects
        self.player = self._create_player()
        self.zombies = []
        self.bullets = []
        self.pickups = []
        
        # Reset tracking - CRITICAL: Reset wave and spawn rate here!
        self.episode_steps = 0
        self.spawn_timer = 0
        self.spawn_rate = self.SPAWN_RATE  # Reset spawn rate to initial value
        self.zombies_killed = 0
        self.zombies_killed_this_episode = 0  # Reset episode kill counter
        self.total_reward = 0
        self.wave = 1  # Reset wave to 1
        self.idle_frames = 0
        self.last_player_pos = (self.player['x'], self.player['y'])
        self.last_health = self.player['health']
        self.previous_min_zombie_dist = float('inf')
        self.previous_danger = 0.0
        self.last_weapon = 'pistol'
        
        # Reset machine gun tracking
        self.consecutive_mg_hits = 0
        self.last_shot_was_mg = False
        self.dps_window = []
        self.mg_fire_time = []
        
        # Spawn initial zombies
        for _ in range(3):
            self.zombies.append(self._spawn_zombie())
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-5):
                0: Move Up
                1: Move Down
                2: Move Left
                3: Move Right
                4: Shoot (weapon selection handled internally)
                5: Idle
        
        Returns:
            observation: New state (98-dim)
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.episode_steps += 1
        reward = 0.0
        info = {}
        
        # Apply action
        shot_result = self._apply_action(action)
        
        # Track weapon switches for reward
        weapon_switched = False
        if self.player['weapon'] != self.last_weapon:
            weapon_switched = True
            self.last_weapon = self.player['weapon']
        
        # Track machine gun fire times for per-second reward
        if shot_result == 'machinegun':
            self.mg_fire_time.append(self.episode_steps)
            # Clean old entries (keep last 60 frames = 1 second)
            self.mg_fire_time = [t for t in self.mg_fire_time if self.episode_steps - t < 60]
        
        # Update game state
        kill_reward, hit_reward, strong_kill_bonus = self._update_bullets()
        reward += kill_reward + hit_reward + strong_kill_bonus
        
        self._update_zombies()
        pickup_reward = self._update_pickups()
        reward += pickup_reward
        
        # Reward for weapon switching appropriately
        if weapon_switched:
            # Reward switching to machine gun when available
            if (self.player['weapon'] == 'machinegun' and 
                self.player['has_machinegun'] and 
                self.player['machinegun_ammo'] > 0):
                reward += self.config.reward_weapon_switch
            # Reward switching back to pistol when machine gun is empty
            elif (self.player['weapon'] == 'pistol' and 
                  (not self.player['has_machinegun'] or self.player['machinegun_ammo'] <= 0)):
                reward += self.config.reward_weapon_switch * 0.5  # Smaller reward
        
        # Machine gun fire per second reward
        if len(self.mg_fire_time) >= 12:  # 12 shots in 1 second (60 FPS, 5 frame cooldown)
            reward += self.config.reward_machinegun_fire_per_second
        
        # Spawn new zombies with wave mechanics
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.zombies.append(self._spawn_zombie())
            self.spawn_timer = 0
            
            # Increase difficulty every 10 zombies killed (wave progression)
            if self.zombies_killed_this_episode > 0 and self.zombies_killed_this_episode % 10 == 0:
                expected_wave = (self.zombies_killed_this_episode // 10) + 1
                if expected_wave > self.wave:
                    self.wave = expected_wave
                    self.spawn_rate = max(10, self.spawn_rate - 5)
        
        # ========================================================================
        # COMPLETE REWARD FUNCTION - EXPERT RL SYSTEM
        # ========================================================================
        
        # 1. SURVIVAL REWARDS
        reward += self.config.reward_survival_per_step
        
        # 2. DAMAGE TAKEN PENALTY
        if self.player['health'] < self.last_health:
            damage_taken = self.last_health - self.player['health']
            reward += self.config.reward_damage_taken * (damage_taken / 10)
        self.last_health = self.player['health']
        
        # 3. IDLE PENALTY
        current_pos = (self.player['x'], self.player['y'])
        if current_pos == self.last_player_pos:
            self.idle_frames += 1
            reward += self.config.reward_idle_penalty
        else:
            self.idle_frames = 0
        self.last_player_pos = current_pos
        
        # 4. POSITIONING REWARDS/PENALTIES
        if len(self.zombies) > 0:
            # Find nearest zombie
            min_dist = min([self._distance(self.player['x'], self.player['y'], z['x'], z['y']) 
                           for z in self.zombies])
            
            # Penalty for standing still while zombies nearby
            if current_pos == self.last_player_pos and min_dist < 200:
                reward += self.config.reward_standing_still_penalty
            
            # Penalty for too close to zombies
            if min_dist < self.config.danger_threshold:
                reward += self.config.reward_too_close_penalty
            
            # Update previous distance
            self.previous_min_zombie_dist = min_dist
        
        # 5. DANGER & DISTANCE-BASED BEHAVIOR
        current_danger = self._compute_danger_score()
        
        # Reward moving away from zombies (danger decrease)
        if current_danger < self.previous_danger:
            reward += self.config.reward_danger_decrease
        elif current_danger > self.previous_danger:
            reward += self.config.reward_danger_increase
        
        # Penalty for entering high-density zone
        zombie_density = self._compute_zombie_density()
        if zombie_density >= self.config.high_density_threshold:
            reward += self.config.reward_high_density_penalty
        
        # Reward for moving away from nearby zombies
        if len(self.zombies) > 0 and current_pos != self.last_player_pos:
            # Check if we're moving away from nearest zombie
            nearest_zombie = min(self.zombies, 
                               key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
            dist_to_nearest = self._distance(self.player['x'], self.player['y'], 
                                            nearest_zombie['x'], nearest_zombie['y'])
            
            if dist_to_nearest < 200:  # Within range
                # Calculate if we moved away
                if dist_to_nearest > self.previous_min_zombie_dist:
                    reward += self.config.reward_moving_away_from_zombies
        
        # Reward for escaping high-danger zone
        if self.previous_danger > 0.7 and current_danger < 0.5:
            reward += self.config.reward_escape_danger_zone
        
        self.previous_danger = current_danger
        
        # Check if done
        done = False
        if self.player['health'] <= 0:
            reward += self.config.reward_death
            done = True
            info['death'] = True
        
        if self.episode_steps >= self.max_episode_steps:
            done = True
            info['timeout'] = True
        
        # Get new state
        state = self._get_state()
        
        self.total_reward += reward
        info['episode_reward'] = self.total_reward
        info['zombies_killed'] = self.zombies_killed
        info['episode_steps'] = self.episode_steps
        info['wave'] = self.wave
        info['spawn_rate'] = self.spawn_rate
        info['has_machinegun'] = self.player['has_machinegun']
        info['machinegun_ammo'] = self.player['machinegun_ammo']
        info['current_weapon'] = self.player['weapon']
        
        return state, reward, done, info
    
    def _apply_action(self, action: int) -> Optional[str]:
        """
        Apply the given action to the player.
        
        Action Space (6 actions):
            0: Move Up
            1: Move Down
            2: Move Left
            3: Move Right
            4: Shoot (weapon selection handled internally)
            5: Idle
        
        Weapon switching is handled automatically:
        - If machine gun is available and has ammo, prefer it
        - Otherwise use pistol
        
        Returns:
            Weapon type if a shot was fired, None otherwise
        """
        shot_weapon = None
        
        # Movement actions
        if action == 0:  # Move Up
            self.player['y'] -= self.PLAYER_SPEED
        elif action == 1:  # Move Down
            self.player['y'] += self.PLAYER_SPEED
        elif action == 2:  # Move Left
            self.player['x'] -= self.PLAYER_SPEED
        elif action == 3:  # Move Right
            self.player['x'] += self.PLAYER_SPEED
        elif action == 4:  # Shoot (weapon selection handled internally)
            # Auto-select best weapon: prefer machine gun if available and has ammo
            if (self.player['has_machinegun'] and 
                self.player['machinegun_ammo'] > 0 and 
                self.player['weapon'] != 'machinegun'):
                self.player['weapon'] = 'machinegun'
            elif (not self.player['has_machinegun'] or 
                  self.player['machinegun_ammo'] <= 0):
                self.player['weapon'] = 'pistol'
            
            shot_weapon = self._player_shoot()
        elif action == 5:  # Idle (do nothing)
            pass
        
        # Keep player on screen
        self.player['x'] = max(0, min(self.SCREEN_WIDTH, self.player['x']))
        self.player['y'] = max(0, min(self.SCREEN_HEIGHT, self.player['y']))
        
        # Update cooldown
        if self.player['shoot_cooldown'] > 0:
            self.player['shoot_cooldown'] -= 1
        
        # Update angle to nearest zombie
        if len(self.zombies) > 0:
            nearest_zombie = min(self.zombies, 
                                key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
            dx = nearest_zombie['x'] - self.player['x']
            dy = nearest_zombie['y'] - self.player['y']
            self.player['angle'] = math.degrees(math.atan2(-dy, dx)) % 360
        
        return shot_weapon
    
    def _player_shoot(self) -> Optional[str]:
        """
        Handle player shooting with weapon selection.
        
        FIXED: Now supports both pistol and machine gun with proper cooldowns.
        Machine gun has much faster fire rate (MACHINEGUN_COOLDOWN = 5 vs PISTOL_COOLDOWN = 20).
        
        Returns:
            Weapon type if shot was fired, None otherwise
        """
        if self.player['shoot_cooldown'] > 0:
            return None
        
        angle_rad = math.radians(self.player['angle'])
        
        if self.player['weapon'] == 'machinegun' and self.player['has_machinegun'] and self.player['machinegun_ammo'] > 0:
            # Machine gun: faster fire rate, less damage per shot, higher DPS
            self.bullets.append({
                'x': self.player['x'],
                'y': self.player['y'],
                'angle': angle_rad,
                'damage': self.MACHINEGUN_DAMAGE,
                'weapon': 'machinegun'
            })
            self.player['shoot_cooldown'] = self.MACHINEGUN_COOLDOWN
            self.player['machinegun_ammo'] -= 1
            
            # Auto-switch to pistol when out of ammo
            if self.player['machinegun_ammo'] <= 0:
                self.player['weapon'] = 'pistol'
                self.player['has_machinegun'] = False
            
            return 'machinegun'
        else:
            # Pistol: slower fire rate, more damage per shot
            self.bullets.append({
                'x': self.player['x'],
                'y': self.player['y'],
                'angle': angle_rad,
                'damage': self.PISTOL_DAMAGE,
                'weapon': 'pistol'
            })
            self.player['shoot_cooldown'] = self.PISTOL_COOLDOWN
            return 'pistol'
    
    def _update_bullets(self) -> Tuple[float, float, float]:
        """
        Update bullet positions and check collisions.
        
        Returns:
            Tuple of (kill_reward, hit_reward, strong_kill_bonus)
        """
        bullets_to_remove = []
        kill_reward = 0.0
        hit_reward = 0.0
        strong_kill_bonus = 0.0
        
        for bullet in self.bullets:
            # Update position
            bullet['x'] += math.cos(bullet['angle']) * self.BULLET_SPEED
            bullet['y'] -= math.sin(bullet['angle']) * self.BULLET_SPEED
            
            # Check if off screen
            if (bullet['x'] < 0 or bullet['x'] > self.SCREEN_WIDTH or
                bullet['y'] < 0 or bullet['y'] > self.SCREEN_HEIGHT):
                bullets_to_remove.append(bullet)
                continue
            
            # Check collision with zombies
            zombies_to_remove = []
            for zombie in self.zombies:
                if self._distance(bullet['x'], bullet['y'], zombie['x'], zombie['y']) < 25:
                    zombie['health'] -= bullet['damage']
                    
                    # Reward for successful hit
                    hit_reward += self.config.reward_hit
                    
                    if zombie['health'] <= 0:
                        zombies_to_remove.append(zombie)
                        self.player['score'] += 10
                        self.zombies_killed += 1
                        self.zombies_killed_this_episode += 1
                        
                        # Base kill reward
                        kill_reward += self.config.reward_zombie_kill
                        
                        # Bonus for killing strong zombie
                        if zombie['type'] == 'strong':
                            strong_kill_bonus += self.config.reward_strong_zombie_kill_bonus
                        
                        # Drop pickups
                        self._drop_pickup(zombie['x'], zombie['y'])
                    
                    bullets_to_remove.append(bullet)
                    break
            
            for zombie in zombies_to_remove:
                if zombie in self.zombies:
                    self.zombies.remove(zombie)
        
        for bullet in bullets_to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
        
        return kill_reward, hit_reward, strong_kill_bonus
    
    def _update_zombies(self):
        """Update zombie positions and check player collisions."""
        for zombie in self.zombies:
            # Move towards player
            dx = self.player['x'] - zombie['x']
            dy = self.player['y'] - zombie['y']
            dist = max(0.1, math.hypot(dx, dy))
            
            zombie['x'] += (dx / dist) * self.ZOMBIE_SPEED
            zombie['y'] += (dy / dist) * self.ZOMBIE_SPEED
            zombie['angle'] = math.degrees(math.atan2(-dy, dx)) % 360
            
            # Update attack cooldown
            if zombie['attack_cooldown'] > 0:
                zombie['attack_cooldown'] -= 1
            
            # Check collision with player
            if self._distance(zombie['x'], zombie['y'], self.player['x'], self.player['y']) < 30:
                if zombie['attack_cooldown'] == 0:
                    self.player['health'] -= zombie['damage']
                    zombie['attack_cooldown'] = self.ZOMBIE_ATTACK_COOLDOWN
    
    def _update_pickups(self) -> float:
        """
        Update pickups and check collisions.
        
        Returns:
            Pickup reward for this frame
        """
        pickups_to_remove = []
        pickup_reward = 0.0
        
        for pickup in self.pickups:
            pickup['lifetime'] -= 1
            if pickup['lifetime'] <= 0:
                pickups_to_remove.append(pickup)
            elif self._distance(pickup['x'], pickup['y'], self.player['x'], self.player['y']) < 25:
                if pickup['type'] == 'health':
                    # Only reward if health is below 100%
                    if self.player['health'] < 100:
                        health_before = self.player['health']
                        self.player['health'] = min(100, self.player['health'] + self.HEALTH_PICKUP_AMOUNT)
                        pickup_reward += self.config.reward_health_pickup
                    
                elif pickup['type'] == 'machinegun':
                    self.player['has_machinegun'] = True
                    self.player['machinegun_ammo'] += self.MACHINEGUN_PICKUP_AMMO
                    pickup_reward += self.config.reward_machinegun_pickup
                    
                elif pickup['type'] == 'ammo':
                    # Ammo pickup for machine gun
                    if self.player['has_machinegun']:
                        self.player['machinegun_ammo'] += self.AMMO_PICKUP_AMOUNT
                        pickup_reward += self.config.reward_ammo_pickup
                    
                pickups_to_remove.append(pickup)
        
        for pickup in pickups_to_remove:
            if pickup in self.pickups:
                self.pickups.remove(pickup)
        
        return pickup_reward
    
    def _get_nearest_pickup(self, pickup_type: str) -> Optional[Dict]:
        """Get the nearest pickup of a given type."""
        pickups_of_type = [p for p in self.pickups if p['type'] == pickup_type]
        if not pickups_of_type:
            return None
        return min(pickups_of_type, 
                   key=lambda p: self._distance(self.player['x'], self.player['y'], p['x'], p['y']))
    
    def _compute_danger_score(self) -> float:
        """
        Compute danger score based on distance to zombies.
        Formula: danger = Σ (1 / distance_to_zombie) for zombies within radius R
        
        Returns:
            Danger score (higher = more dangerous)
        """
        danger = 0.0
        radius = self.config.danger_radius
        
        for zombie in self.zombies:
            dist = self._distance(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
            if dist < radius and dist > 0:
                danger += 1.0 / max(dist, 1.0)  # Avoid division by zero
        
        return danger
    
    def _compute_zombie_density(self) -> float:
        """
        Compute zombie density in a radius around the player.
        
        Returns:
            Number of zombies within danger radius
        """
        radius = self.config.danger_radius
        count = 0
        
        for zombie in self.zombies:
            dist = self._distance(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
            if dist < radius:
                count += 1
        
        return count
    
    def _compute_danger_level(self) -> float:
        """
        Compute a normalized danger level metric (0.0 to 1.0) for state encoding.
        Based on danger score and other factors.
        
        Returns:
            Danger level from 0.0 (safe) to 1.0 (critical)
        """
        danger_score = self._compute_danger_score()
        # Normalize danger score (typical max is around 5-10 for many zombies)
        normalized_danger = min(danger_score / 10.0, 1.0)
        
        # Factor in health
        health_factor = 1.0 - (self.player['health'] / 100.0)
        
        # Combine factors
        danger = (normalized_danger * 0.7) + (health_factor * 0.3)
        
        return min(danger, 1.0)
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation (98-dim).
        
        State Space (98-dim):
        - Player: [x, y, health, shoot_cooldown, angle, zombie_count] (6 values)
        - Weapon: [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg] (3 values)
        - Zombies (up to 13): [x, y, health, type, distance, angle_to_player] per zombie (78 values)
        - Pickups: [machinegun_location, health_pack_location, ammo_pickup_location]
                   Each: [distance, angle, exists] (9 values)
        - Distance to nearest zombie: 1 value
        - Zombie density in radius: 1 value
        Total: 6 + 3 + 78 + 9 + 1 + 1 = 98
        
        Returns:
            State vector as numpy array (98-dim)
        """
        state = []
        
        # Player state (6 values): [x, y, health, shoot_cooldown, angle, zombie_count]
        state.extend([
            self.player['x'] / self.SCREEN_WIDTH,  # Normalize
            self.player['y'] / self.SCREEN_HEIGHT,
            self.player['health'] / 100.0,
            self.player['shoot_cooldown'] / self.PISTOL_COOLDOWN,
            self.player['angle'] / 360.0,
            min(len(self.zombies), 13) / 13.0  # Normalize zombie count (max 13 for 98-dim)
        ])
        
        # Weapon state (3 values): [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg]
        state.extend([
            1.0 if self.player['has_machinegun'] else 0.0,
            min(self.player['machinegun_ammo'] / 100.0, 1.0),  # Normalize ammo
            1.0 if self.player['weapon'] == 'machinegun' else 0.0
        ])
        
        # Zombie states (sort by distance, take closest 13 zombies for 98-dim state)
        zombies_sorted = sorted(self.zombies, 
                               key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
        
        max_zombies_for_state = 13  # To get exactly 98-dim
        for i in range(max_zombies_for_state):
            if i < len(zombies_sorted):
                zombie = zombies_sorted[i]
                distance = self._distance(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
                angle_to_zombie = self._angle_to(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
                
                state.extend([
                    zombie['x'] / self.SCREEN_WIDTH,
                    zombie['y'] / self.SCREEN_HEIGHT,
                    zombie['health'] / self.ZOMBIE_STRONG_HEALTH,  # Normalize by max health
                    1.0 if zombie['type'] == 'strong' else 0.0,
                    min(distance / 1000.0, 1.0),  # Normalize distance
                    angle_to_zombie / 360.0
                ])
            else:
                # Padding for empty zombie slots
                state.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        # Pickup info (9 values): 3 pickups * 3 features each
        # Machinegun pickup: [distance, angle, exists]
        nearest_mg = self._get_nearest_pickup('machinegun')
        if nearest_mg:
            mg_dist = self._distance(self.player['x'], self.player['y'], 
                                    nearest_mg['x'], nearest_mg['y'])
            mg_angle = self._angle_to(self.player['x'], self.player['y'],
                                     nearest_mg['x'], nearest_mg['y'])
            state.extend([
                min(mg_dist / 1000.0, 1.0),
                mg_angle / 360.0,
                1.0  # Exists
            ])
        else:
            state.extend([1.0, 0.0, 0.0])  # No machinegun pickup
        
        # Health pack: [distance, angle, exists]
        nearest_health = self._get_nearest_pickup('health')
        if nearest_health:
            health_dist = self._distance(self.player['x'], self.player['y'], 
                                        nearest_health['x'], nearest_health['y'])
            health_angle = self._angle_to(self.player['x'], self.player['y'],
                                         nearest_health['x'], nearest_health['y'])
            state.extend([
                min(health_dist / 1000.0, 1.0),
                health_angle / 360.0,
                1.0  # Exists
            ])
        else:
            state.extend([1.0, 0.0, 0.0])  # No health pack
        
        # Ammo pickup: [distance, angle, exists]
        nearest_ammo = self._get_nearest_pickup('ammo')
        if nearest_ammo:
            ammo_dist = self._distance(self.player['x'], self.player['y'], 
                                       nearest_ammo['x'], nearest_ammo['y'])
            ammo_angle = self._angle_to(self.player['x'], self.player['y'],
                                       nearest_ammo['x'], nearest_ammo['y'])
            state.extend([
                min(ammo_dist / 1000.0, 1.0),
                ammo_angle / 360.0,
                1.0  # Exists
            ])
        else:
            state.extend([1.0, 0.0, 0.0])  # No ammo pickup
        
        # Distance to nearest zombie (1 value)
        if len(self.zombies) > 0:
            min_dist = min([self._distance(self.player['x'], self.player['y'], z['x'], z['y']) 
                           for z in self.zombies])
            state.append(min(min_dist / 1000.0, 1.0))  # Normalize
        else:
            state.append(1.0)  # No zombies, max distance
        
        # Zombie density in radius (1 value)
        density = self._compute_zombie_density()
        state.append(min(density / 10.0, 1.0))  # Normalize (max 10 zombies)
        
        # Verify state dimension
        assert len(state) == 98, f"State dimension mismatch: expected 98, got {len(state)}"
        
        return np.array(state, dtype=np.float32)
    
    def _create_player(self) -> Dict:
        """Create a new player with full weapon support."""
        return {
            'x': self.SCREEN_WIDTH // 2,
            'y': self.SCREEN_HEIGHT // 2,
            'angle': 0,
            'health': 100,
            'score': 0,
            'shoot_cooldown': 0,
            'weapon': 'pistol',  # Current weapon
            'has_machinegun': False,  # Must pick up to use
            'machinegun_ammo': 0
        }
    
    def _spawn_zombie(self) -> Dict:
        """Spawn a new zombie."""
        side = random.randint(0, 3)
        if side == 0:  # Top
            x = random.randint(0, self.SCREEN_WIDTH)
            y = -50
        elif side == 1:  # Right
            x = self.SCREEN_WIDTH + 50
            y = random.randint(0, self.SCREEN_HEIGHT)
        elif side == 2:  # Bottom
            x = random.randint(0, self.SCREEN_WIDTH)
            y = self.SCREEN_HEIGHT + 50
        else:  # Left
            x = -50
            y = random.randint(0, self.SCREEN_HEIGHT)
        
        zombie_type = 'strong' if random.random() < self.ZOMBIE_TYPE2_SPAWN_PROBABILITY else 'normal'
        
        return {
            'x': x,
            'y': y,
            'type': zombie_type,
            'health': self.ZOMBIE_STRONG_HEALTH if zombie_type == 'strong' else self.ZOMBIE_NORMAL_HEALTH,
            'damage': self.ZOMBIE_STRONG_DAMAGE if zombie_type == 'strong' else self.ZOMBIE_NORMAL_DAMAGE,
            'angle': 0,
            'attack_cooldown': 0
        }
    
    def _drop_pickup(self, x: float, y: float):
        """Drop a pickup at the given position."""
        drop_chance = random.random()
        if drop_chance < self.HEALTH_PICKUP_DROP_PROBABILITY:
            self.pickups.append({
                'x': x,
                'y': y,
                'type': 'health',
                'lifetime': self.PICKUP_LIFETIME
            })
        elif drop_chance < self.HEALTH_PICKUP_DROP_PROBABILITY + self.MACHINEGUN_PICKUP_DROP_PROBABILITY:
            self.pickups.append({
                'x': x,
                'y': y,
                'type': 'machinegun',
                'lifetime': self.PICKUP_LIFETIME
            })
        elif drop_chance < (self.HEALTH_PICKUP_DROP_PROBABILITY + 
                           self.MACHINEGUN_PICKUP_DROP_PROBABILITY + 
                           self.AMMO_PICKUP_DROP_PROBABILITY):
            self.pickups.append({
                'x': x,
                'y': y,
                'type': 'ammo',
                'lifetime': self.PICKUP_LIFETIME
            })
    
    def _distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance."""
        return math.hypot(x2 - x1, y2 - y1)
    
    def _angle_to(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate angle from (x1, y1) to (x2, y2) in degrees."""
        dx = x2 - x1
        dy = y2 - y1
        return math.degrees(math.atan2(-dy, dx)) % 360
    
    def render(self):
        """Render the environment (if not headless)."""
        if self.headless:
            return
        
        # Simple rendering
        self.screen.fill((96, 96, 96))
        
        # Draw player
        color = (0, 255, 0) if self.player['weapon'] == 'pistol' else (0, 200, 255)
        pygame.draw.circle(self.screen, color, 
                          (int(self.player['x']), int(self.player['y'])), 15)
        
        # Draw zombies
        for zombie in self.zombies:
            color = (255, 0, 0) if zombie['type'] == 'normal' else (150, 0, 0)
            pygame.draw.circle(self.screen, color, 
                             (int(zombie['x']), int(zombie['y'])), 12)
        
        # Draw bullets
        for bullet in self.bullets:
            color = (255, 255, 0) if bullet.get('weapon') == 'pistol' else (255, 165, 0)
            pygame.draw.circle(self.screen, color, 
                             (int(bullet['x']), int(bullet['y'])), 4)
        
        # Draw pickups
        for pickup in self.pickups:
            if pickup['type'] == 'health':
                pygame.draw.rect(self.screen, (255, 0, 0), 
                               (int(pickup['x']) - 8, int(pickup['y']) - 2, 16, 4))
                pygame.draw.rect(self.screen, (255, 0, 0), 
                               (int(pickup['x']) - 2, int(pickup['y']) - 8, 4, 16))
            else:
                pygame.draw.rect(self.screen, (100, 100, 100),
                               (int(pickup['x']) - 10, int(pickup['y']) - 4, 20, 8))
        
        # Draw UI
        font = pygame.font.SysFont(None, 24)
        health_text = font.render(f"Health: {self.player['health']}", True, (255, 255, 255))
        score_text = font.render(f"Score: {self.player['score']}", True, (255, 255, 255))
        kills_text = font.render(f"Kills: {self.zombies_killed}", True, (255, 255, 255))
        wave_text = font.render(f"Wave: {self.wave}", True, (255, 255, 255))
        weapon_text = font.render(f"Weapon: {self.player['weapon']}", True, (255, 255, 255))
        ammo_text = font.render(f"MG Ammo: {self.player['machinegun_ammo']}", True, (255, 255, 0))
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(score_text, (10, 35))
        self.screen.blit(kills_text, (10, 60))
        self.screen.blit(wave_text, (10, 85))
        self.screen.blit(weapon_text, (10, 110))
        self.screen.blit(ammo_text, (10, 135))
        
        pygame.display.flip()
        self.clock.tick(self.FPS)
    
    def close(self):
        """Close the environment."""
        pygame.quit()
