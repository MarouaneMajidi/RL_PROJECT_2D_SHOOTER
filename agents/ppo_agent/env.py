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
    
    EXPERT RL SYSTEM - COMPREHENSIVE REDESIGN FOR OPTIMAL BEHAVIOR:
    
    Action Space (Multi-Discrete):
        Combined action = movement_action * 2 + shoot_action
        - Movement: 0=Up, 1=Down, 2=Left, 3=Right, 4=Idle (5 options)
        - Shoot: 0=Don't shoot, 1=Shoot (2 options)
        - Allows simultaneous movement and shooting!
    
    State Space (111-dim with enhanced features):
        - Player: [x, y, health, shoot_cooldown, angle, zombie_count] (6 values)
        - Weapon: [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg] (3 values)
        - Zombies (up to 13): [x, y, health, type, distance, angle_to_player] per zombie (78 values)
        - Pickups (enhanced with directional info):
          Each pickup: [distance, angle, exists, dx_normalized, dy_normalized, in_range] (6 values)
          Machinegun, Health, Ammo = 18 values total
        - Distance to nearest zombie: 1 value
        - Zombie density in radius: 1 value
        - Movement direction (last action): [up, down, left, right] (4 values)
        Total: 6 + 3 + 78 + 18 + 1 + 1 + 4 = 111
    
    Key Improvements:
    1. Multi-discrete actions enable shooting while moving
    2. Directional pickup rewards encourage active seeking
    3. Enhanced state space with relative positions
    4. Dense reward shaping for combat behavior
    5. Increased pickup collection radius for reliability
    """
    
    # Game constants (imported from main.py)
    # Arena (playable area) - stays exactly the same
    ARENA_WIDTH = 800
    ARENA_HEIGHT = 600
    
    # Expanded screen size (visible area, larger than arena)
    SCREEN_WIDTH = 1000  # Expanded to show area around arena
    SCREEN_HEIGHT = 800  # Expanded to show area around arena
    
    # Glass border configuration
    GLASS_BORDER_THICKNESS = 5  # Thickness of glass border in pixels
    GLASS_COLOR = (200, 200, 255, 180)  # Semi-transparent blue-white glass (RGBA)
    
    # Calculate arena position (centered in expanded screen)
    ARENA_X_OFFSET = (SCREEN_WIDTH - ARENA_WIDTH) // 2
    ARENA_Y_OFFSET = (SCREEN_HEIGHT - ARENA_HEIGHT) // 2
    
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
        self.last_movement_direction = [0, 0, 0, 0]  # Track last movement direction [up, down, left, right]
        
        # Pickup tracking for directional rewards
        self.previous_pickup_distances = {}  # Track previous distances to pickups
        
        # Combat behavior tracking
        self.was_moving = False
        self.was_shooting = False
        self.enemies_in_range = False
        
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
        self.last_movement_direction = [0, 0, 0, 0]
        self.previous_pickup_distances = {}
        self.was_moving = False
        self.was_shooting = False
        self.enemies_in_range = False
        
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
        Execute one step in the environment with multi-discrete actions.
        
        Args:
            action: Combined action (movement * 2 + shoot)
                   Movement: 0=Up, 1=Down, 2=Left, 3=Right, 4=Idle
                   Shoot: 0=Don't shoot, 1=Shoot
        
        Returns:
            observation: New state (111-dim with enhanced features)
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.episode_steps += 1
        reward = 0.0
        info = {}
        
        # Decode multi-discrete action
        movement_action = action // self.config.action_space_shape[1]
        shoot_action = action % self.config.action_space_shape[1]
        
        # Track movement and shooting for rewards
        self.was_moving = (movement_action < 4)  # Not idle
        self.was_shooting = (shoot_action == 1)
        
        # Apply action (movement and shooting simultaneously)
        shot_result = self._apply_action_multi_discrete(movement_action, shoot_action)
        
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
        
        # Check if enemies are in shooting range (only in-bounds zombies)
        self.enemies_in_range = False
        in_bounds_zombies = self._get_in_bounds_zombies()
        if len(in_bounds_zombies) > 0:
            nearest_zombie = min(in_bounds_zombies, 
                               key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
            dist_to_nearest = self._distance(self.player['x'], self.player['y'], 
                                            nearest_zombie['x'], nearest_zombie['y'])
            self.enemies_in_range = (dist_to_nearest < 400)  # Shooting range
        
        # Update game state
        kill_reward, hit_reward, strong_kill_bonus = self._update_bullets()
        reward += kill_reward + hit_reward + strong_kill_bonus
        
        self._update_zombies()
        pickup_reward = self._update_pickups()
        reward += pickup_reward
        
        # NEW: Directional pickup rewards (dense shaping)
        reward += self._compute_directional_pickup_rewards()
        
        # NEW: Combat behavior rewards
        if self.was_shooting and self.was_moving:
            reward += self.config.reward_shooting_while_moving
        if self.was_shooting and self.enemies_in_range:
            reward += self.config.reward_shooting_at_enemies
        if self.was_moving and not self.was_shooting and self.enemies_in_range:
            # Running away without shooting
            reward += self.config.penalty_running_away_without_shooting
        
        # Penalty for shooting when no in-bounds zombies (wasting ammo on off-screen targets)
        if self.was_shooting and len(self._get_in_bounds_zombies()) == 0:
            reward += self.config.penalty_shooting_at_out_of_bounds
        
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
        
        # 4. POSITIONING REWARDS/PENALTIES (only consider in-bounds zombies)
        in_bounds_zombies = self._get_in_bounds_zombies()
        if len(in_bounds_zombies) > 0:
            # Find nearest in-bounds zombie
            min_dist = min([self._distance(self.player['x'], self.player['y'], z['x'], z['y']) 
                           for z in in_bounds_zombies])
            
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
        
        # Reward for moving away from nearby zombies (only in-bounds)
        if len(in_bounds_zombies) > 0 and current_pos != self.last_player_pos:
            # Check if we're moving away from nearest in-bounds zombie
            nearest_zombie = min(in_bounds_zombies, 
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
        
        # NEW: Update pickup distances for next step
        self._update_pickup_distances()
        
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
        info['score'] = self.player['score']  # Game score (increases with zombie kills)
        
        return state, reward, done, info
    
    def _apply_action_multi_discrete(self, movement_action: int, shoot_action: int) -> Optional[str]:
        """
        Apply multi-discrete actions (movement and shooting simultaneously).
        
        Args:
            movement_action: 0=Up, 1=Down, 2=Left, 3=Right, 4=Idle
            shoot_action: 0=Don't shoot, 1=Shoot
        
        Returns:
            Weapon type if a shot was fired, None otherwise
        """
        shot_weapon = None
        
        # Apply movement
        moved = False
        if movement_action == 0:  # Move Up
            self.player['y'] -= self.PLAYER_SPEED
            moved = True
            self.last_movement_direction = [1, 0, 0, 0]
        elif movement_action == 1:  # Move Down
            self.player['y'] += self.PLAYER_SPEED
            moved = True
            self.last_movement_direction = [0, 1, 0, 0]
        elif movement_action == 2:  # Move Left
            self.player['x'] -= self.PLAYER_SPEED
            moved = True
            self.last_movement_direction = [0, 0, 1, 0]
        elif movement_action == 3:  # Move Right
            self.player['x'] += self.PLAYER_SPEED
            moved = True
            self.last_movement_direction = [0, 0, 0, 1]
        elif movement_action == 4:  # Idle
            self.last_movement_direction = [0, 0, 0, 0]
        
        # Apply shooting (independent of movement)
        if shoot_action == 1:  # Shoot
            # Auto-select best weapon: prefer machine gun if available and has ammo
            if (self.player['has_machinegun'] and 
                self.player['machinegun_ammo'] > 0 and 
                self.player['weapon'] != 'machinegun'):
                self.player['weapon'] = 'machinegun'
            elif (not self.player['has_machinegun'] or 
                  self.player['machinegun_ammo'] <= 0):
                self.player['weapon'] = 'pistol'
            
            shot_weapon = self._player_shoot()
        
        # Keep player within arena bounds (glass borders prevent movement outside)
        # Arena is positioned at (ARENA_X_OFFSET, ARENA_Y_OFFSET) in the expanded screen
        self.player['x'] = max(self.ARENA_X_OFFSET, 
                               min(self.ARENA_X_OFFSET + self.ARENA_WIDTH, self.player['x']))
        self.player['y'] = max(self.ARENA_Y_OFFSET, 
                               min(self.ARENA_Y_OFFSET + self.ARENA_HEIGHT, self.player['y']))
        
        # Update cooldown
        if self.player['shoot_cooldown'] > 0:
            self.player['shoot_cooldown'] -= 1
        
        # Update angle to nearest IN-BOUNDS zombie only
        in_bounds_zombies = self._get_in_bounds_zombies()
        if len(in_bounds_zombies) > 0:
            nearest_zombie = min(in_bounds_zombies, 
                                key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
            dx = nearest_zombie['x'] - self.player['x']
            dy = nearest_zombie['y'] - self.player['y']
            self.player['angle'] = math.degrees(math.atan2(-dy, dx)) % 360
        # If no in-bounds zombies, keep previous angle
        
        return shot_weapon
    
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
        
        # Keep player within arena bounds (glass borders prevent movement outside)
        # Arena is positioned at (ARENA_X_OFFSET, ARENA_Y_OFFSET) in the expanded screen
        self.player['x'] = max(self.ARENA_X_OFFSET, 
                               min(self.ARENA_X_OFFSET + self.ARENA_WIDTH, self.player['x']))
        self.player['y'] = max(self.ARENA_Y_OFFSET, 
                               min(self.ARENA_Y_OFFSET + self.ARENA_HEIGHT, self.player['y']))
        
        # Update cooldown
        if self.player['shoot_cooldown'] > 0:
            self.player['shoot_cooldown'] -= 1
        
        # Update angle to nearest IN-BOUNDS zombie only
        in_bounds_zombies = self._get_in_bounds_zombies()
        if len(in_bounds_zombies) > 0:
            nearest_zombie = min(in_bounds_zombies, 
                                key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
            dx = nearest_zombie['x'] - self.player['x']
            dy = nearest_zombie['y'] - self.player['y']
            self.player['angle'] = math.degrees(math.atan2(-dy, dx)) % 360
        else:
            # No in-bounds zombies, don't update angle
            pass
        
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
            
            # Check if off screen (expanded screen bounds)
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
        Update pickups and check collisions with strategic reward scaling.
        
        Returns:
            Pickup reward for this frame (scaled by urgency/need)
        """
        pickups_to_remove = []
        pickup_reward = 0.0
        urgency = self._compute_pickup_urgency()
        
        for pickup in self.pickups:
            pickup['lifetime'] -= 1
            if pickup['lifetime'] <= 0:
                pickups_to_remove.append(pickup)
            elif self._distance(pickup['x'], pickup['y'], self.player['x'], self.player['y']) < self.config.pickup_collection_radius:
                if pickup['type'] == 'health':
                    # Strategic reward: higher when health is low (more urgent)
                    health_before = self.player['health']
                    self.player['health'] = min(100, self.player['health'] + self.HEALTH_PICKUP_AMOUNT)
                    
                    # Only give reward if health was actually increased
                    if health_before < 100:
                        # Scale reward by urgency (low health = higher reward)
                        base_reward = self.config.reward_health_pickup
                        urgency_multiplier = 1.0 + urgency['health'] * 2.0  # 1x to 3x
                        pickup_reward += base_reward * urgency_multiplier
                        
                        # Strategic bonus when health is low
                        if urgency['health'] > 0.5:  # More than 50% urgency
                            pickup_reward += self.config.reward_strategic_health_pickup
                    
                elif pickup['type'] == 'machinegun':
                    # Machinegun is always valuable (dramatically increases lethality)
                    self.player['has_machinegun'] = True
                    self.player['machinegun_ammo'] += self.MACHINEGUN_PICKUP_AMMO
                    
                    # Higher reward if we don't have it (urgency = 1.0)
                    base_reward = self.config.reward_machinegun_pickup
                    urgency_multiplier = 1.0 + urgency['machinegun'] * 1.5  # 1x to 2.5x
                    pickup_reward += base_reward * urgency_multiplier
                    pickup_reward += self.config.reward_strategic_machinegun_pickup  # Strategic bonus
                    
                elif pickup['type'] == 'ammo':
                    # Ammo pickup for machine gun (only valuable if have machinegun)
                    if self.player['has_machinegun']:
                        self.player['machinegun_ammo'] += self.AMMO_PICKUP_AMOUNT
                        
                        # Scale reward by ammo urgency (low ammo = higher reward)
                        base_reward = self.config.reward_ammo_pickup
                        urgency_multiplier = 1.0 + urgency['ammo'] * 1.5  # 1x to 2.5x
                        pickup_reward += base_reward * urgency_multiplier
                        
                        # Strategic bonus when ammo is low
                        if urgency['ammo'] > 0.5:  # More than 50% urgency
                            pickup_reward += self.config.reward_strategic_ammo_pickup
                    # No reward if don't have machinegun (urgency = 0.0)
                    
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
        Compute danger score based on distance to IN-BOUNDS zombies only.
        Formula: danger = Σ (1 / distance_to_zombie) for zombies within radius R
        
        Returns:
            Danger score (higher = more dangerous)
        """
        danger = 0.0
        radius = self.config.danger_radius
        in_bounds_zombies = self._get_in_bounds_zombies()
        
        for zombie in in_bounds_zombies:
            dist = self._distance(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
            if dist < radius and dist > 0:
                danger += 1.0 / max(dist, 1.0)  # Avoid division by zero
        
        return danger
    
    def _compute_zombie_density(self) -> float:
        """
        Compute zombie density in a radius around the player (only in-bounds zombies).
        
        Returns:
            Number of in-bounds zombies within danger radius
        """
        radius = self.config.danger_radius
        count = 0
        in_bounds_zombies = self._get_in_bounds_zombies()
        
        for zombie in in_bounds_zombies:
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
    
    def _compute_directional_pickup_rewards(self) -> float:
        """
        Compute directional rewards for moving toward pickups with urgency weighting.
        
        Returns:
            Total directional pickup reward
        """
        reward = 0.0
        urgency = self._compute_pickup_urgency()
        
        # Check each pickup type
        for pickup_type in ['machinegun', 'health', 'ammo']:
            nearest = self._get_nearest_pickup(pickup_type)
            if nearest is None:
                continue
            
            current_dist = self._distance(self.player['x'], self.player['y'], 
                                         nearest['x'], nearest['y'])
            
            # Get previous distance
            key = f"{pickup_type}_distance"
            if key in self.previous_pickup_distances:
                prev_dist = self.previous_pickup_distances[key]
                
                # Reward moving closer, weighted by urgency
                if current_dist < prev_dist:
                    base_reward = 0.0
                    if pickup_type == 'machinegun':
                        base_reward = self.config.reward_moving_toward_machinegun
                    elif pickup_type == 'health':
                        base_reward = self.config.reward_moving_toward_health
                    elif pickup_type == 'ammo':
                        base_reward = self.config.reward_moving_toward_ammo
                    
                    # Scale reward by urgency (more urgent = higher reward)
                    urgency_weight = urgency[pickup_type]
                    reward += base_reward * (1.0 + urgency_weight * 2.0)  # 1x to 3x multiplier
                
                # Bonus for being in pickup range, weighted by urgency
                if current_dist < self.config.pickup_detection_range:
                    proximity_reward = self.config.reward_pickup_proximity * (1.0 + urgency[pickup_type])
                    reward += proximity_reward
        
        return reward
    
    def _is_zombie_in_bounds(self, zombie: Dict) -> bool:
        """
        Check if a zombie is within the playable arena area (inside glass borders).
        
        Args:
            zombie: Zombie dictionary with 'x' and 'y' keys
        
        Returns:
            True if zombie is in bounds (inside arena), False otherwise
        """
        return (self.ARENA_X_OFFSET <= zombie['x'] <= self.ARENA_X_OFFSET + self.ARENA_WIDTH and 
                self.ARENA_Y_OFFSET <= zombie['y'] <= self.ARENA_Y_OFFSET + self.ARENA_HEIGHT)
    
    def _get_in_bounds_zombies(self) -> List[Dict]:
        """
        Get only zombies that are within the playable area.
        
        Returns:
            List of in-bounds zombies
        """
        return [z for z in self.zombies if self._is_zombie_in_bounds(z)]
    
    def _compute_pickup_urgency(self) -> Dict[str, float]:
        """
        Compute urgency scores for each pickup type based on current state.
        
        Returns:
            Dictionary with urgency scores (0.0 to 1.0) for each pickup type
        """
        urgency = {
            'health': 0.0,
            'machinegun': 0.0,
            'ammo': 0.0
        }
        
        # Health urgency: increases as health decreases
        health_ratio = self.player['health'] / 100.0
        urgency['health'] = 1.0 - health_ratio  # 1.0 when health = 0, 0.0 when health = 100
        
        # Machinegun urgency: high if don't have it, low if have it
        if not self.player['has_machinegun']:
            urgency['machinegun'] = 1.0  # Critical: need machinegun
        else:
            urgency['machinegun'] = 0.0  # Already have it
        
        # Ammo urgency: increases as ammo decreases (only if have machinegun)
        if self.player['has_machinegun']:
            ammo_ratio = self.player['machinegun_ammo'] / 100.0
            urgency['ammo'] = 1.0 - ammo_ratio  # 1.0 when ammo = 0, 0.0 when ammo = 100
        else:
            urgency['ammo'] = 0.0  # No urgency if don't have machinegun
        
        return urgency
    
    def _update_pickup_distances(self):
        """Update pickup distances for next step's directional rewards."""
        for pickup_type in ['machinegun', 'health', 'ammo']:
            nearest = self._get_nearest_pickup(pickup_type)
            if nearest:
                dist = self._distance(self.player['x'], self.player['y'], 
                                     nearest['x'], nearest['y'])
                self.previous_pickup_distances[f"{pickup_type}_distance"] = dist
            else:
                # Remove if pickup no longer exists
                key = f"{pickup_type}_distance"
                if key in self.previous_pickup_distances:
                    del self.previous_pickup_distances[key]
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation (111-dim with enhanced features).
        
        Enhanced State Space (129-dim):
        - Player: [x, y, health, shoot_cooldown, angle, zombie_count] (6 values)
        - Weapon: [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg] (3 values)
        - Zombies (up to 13): [x, y, health, type, distance, angle_to_player, in_bounds] per zombie (91 values)
          * Prioritizes in-bounds zombies in state encoding
          * in_bounds flag helps agent distinguish valid targets
        - Pickups (enhanced with directional info):
          Machinegun: [distance, angle, exists, dx_normalized, dy_normalized, in_range] (6 values)
          Health: [distance, angle, exists, dx_normalized, dy_normalized, in_range] (6 values)
          Ammo: [distance, angle, exists, dx_normalized, dy_normalized, in_range] (6 values)
          Total: 18 values
        - Distance to nearest IN-BOUNDS zombie: 1 value
        - Zombie density in radius (in-bounds only): 1 value
        - Movement direction (last action): [up, down, left, right] (4 values)
        - Pickup urgency signals: [health_urgency, machinegun_urgency, ammo_urgency] (3 values)
        - Zombie counts: [in_bounds_count_normalized, total_count_normalized] (2 values)
        Total: 6 + 3 + 91 + 18 + 1 + 1 + 4 + 3 + 2 = 129
        
        Returns:
            State vector as numpy array (111-dim)
        """
        state = []
        
        # Player state (6 values): [x, y, health, shoot_cooldown, angle, zombie_count]
        # Use in-bounds zombie count for more accurate representation
        in_bounds_count = len(self._get_in_bounds_zombies())
        # Normalize player position relative to arena (not screen)
        state.extend([
            (self.player['x'] - self.ARENA_X_OFFSET) / self.ARENA_WIDTH,  # Normalize to arena
            (self.player['y'] - self.ARENA_Y_OFFSET) / self.ARENA_HEIGHT,  # Normalize to arena
            self.player['health'] / 100.0,
            self.player['shoot_cooldown'] / self.PISTOL_COOLDOWN,
            self.player['angle'] / 360.0,
            min(in_bounds_count, 13) / 13.0  # Normalize in-bounds zombie count
        ])
        
        # Weapon state (3 values): [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg]
        state.extend([
            1.0 if self.player['has_machinegun'] else 0.0,
            min(self.player['machinegun_ammo'] / 100.0, 1.0),  # Normalize ammo
            1.0 if self.player['weapon'] == 'machinegun' else 0.0
        ])
        
        # Zombie states (sort by distance, prioritize IN-BOUNDS zombies, take closest 13)
        # Separate in-bounds and out-of-bounds zombies
        in_bounds_zombies = self._get_in_bounds_zombies()
        out_of_bounds_zombies = [z for z in self.zombies if not self._is_zombie_in_bounds(z)]
        
        # Prioritize in-bounds zombies, then out-of-bounds
        zombies_sorted = (sorted(in_bounds_zombies, 
                                key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y'])) +
                         sorted(out_of_bounds_zombies,
                                key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y'])))
        
        max_zombies_for_state = 13
        for i in range(max_zombies_for_state):
            if i < len(zombies_sorted):
                zombie = zombies_sorted[i]
                distance = self._distance(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
                angle_to_zombie = self._angle_to(self.player['x'], self.player['y'], zombie['x'], zombie['y'])
                is_in_bounds = 1.0 if self._is_zombie_in_bounds(zombie) else 0.0
                
                # Normalize zombie position relative to arena (for consistency with player)
                state.extend([
                    (zombie['x'] - self.ARENA_X_OFFSET) / self.ARENA_WIDTH,
                    (zombie['y'] - self.ARENA_Y_OFFSET) / self.ARENA_HEIGHT,
                    zombie['health'] / self.ZOMBIE_STRONG_HEALTH,  # Normalize by max health
                    1.0 if zombie['type'] == 'strong' else 0.0,
                    min(distance / 1000.0, 1.0),  # Normalize distance
                    angle_to_zombie / 360.0,
                    is_in_bounds  # NEW: Flag indicating if zombie is in bounds
                ])
            else:
                # Padding for empty zombie slots
                state.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        
        # Pickup info (enhanced with directional info): 18 values total
        # Machinegun pickup: [distance, angle, exists, dx_normalized, dy_normalized, in_range]
        nearest_mg = self._get_nearest_pickup('machinegun')
        if nearest_mg:
            mg_dist = self._distance(self.player['x'], self.player['y'], 
                                    nearest_mg['x'], nearest_mg['y'])
            mg_angle = self._angle_to(self.player['x'], self.player['y'],
                                     nearest_mg['x'], nearest_mg['y'])
            dx = (nearest_mg['x'] - self.player['x']) / self.ARENA_WIDTH
            dy = (nearest_mg['y'] - self.player['y']) / self.ARENA_HEIGHT
            in_range = 1.0 if mg_dist < self.config.pickup_detection_range else 0.0
            state.extend([
                min(mg_dist / 1000.0, 1.0),
                mg_angle / 360.0,
                1.0,  # Exists
                dx,  # Normalized dx
                dy,  # Normalized dy
                in_range
            ])
        else:
            state.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No machinegun pickup
        
        # Health pack: [distance, angle, exists, dx_normalized, dy_normalized, in_range]
        nearest_health = self._get_nearest_pickup('health')
        if nearest_health:
            health_dist = self._distance(self.player['x'], self.player['y'], 
                                        nearest_health['x'], nearest_health['y'])
            health_angle = self._angle_to(self.player['x'], self.player['y'],
                                         nearest_health['x'], nearest_health['y'])
            dx = (nearest_health['x'] - self.player['x']) / self.ARENA_WIDTH
            dy = (nearest_health['y'] - self.player['y']) / self.ARENA_HEIGHT
            in_range = 1.0 if health_dist < self.config.pickup_detection_range else 0.0
            state.extend([
                min(health_dist / 1000.0, 1.0),
                health_angle / 360.0,
                1.0,  # Exists
                dx,  # Normalized dx
                dy,  # Normalized dy
                in_range
            ])
        else:
            state.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No health pack
        
        # Ammo pickup: [distance, angle, exists, dx_normalized, dy_normalized, in_range]
        nearest_ammo = self._get_nearest_pickup('ammo')
        if nearest_ammo:
            ammo_dist = self._distance(self.player['x'], self.player['y'], 
                                       nearest_ammo['x'], nearest_ammo['y'])
            ammo_angle = self._angle_to(self.player['x'], self.player['y'],
                                       nearest_ammo['x'], nearest_ammo['y'])
            dx = (nearest_ammo['x'] - self.player['x']) / self.ARENA_WIDTH
            dy = (nearest_ammo['y'] - self.player['y']) / self.ARENA_HEIGHT
            in_range = 1.0 if ammo_dist < self.config.pickup_detection_range else 0.0
            state.extend([
                min(ammo_dist / 1000.0, 1.0),
                ammo_angle / 360.0,
                1.0,  # Exists
                dx,  # Normalized dx
                dy,  # Normalized dy
                in_range
            ])
        else:
            state.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No ammo pickup
        
        # Distance to nearest IN-BOUNDS zombie (1 value)
        in_bounds_zombies = self._get_in_bounds_zombies()
        if len(in_bounds_zombies) > 0:
            min_dist = min([self._distance(self.player['x'], self.player['y'], z['x'], z['y']) 
                           for z in in_bounds_zombies])
            state.append(min(min_dist / 1000.0, 1.0))  # Normalize
        else:
            state.append(1.0)  # No in-bounds zombies, max distance
        
        # Zombie density in radius (1 value)
        density = self._compute_zombie_density()
        state.append(min(density / 10.0, 1.0))  # Normalize (max 10 zombies)
        
        # Movement direction (4 values): [up, down, left, right]
        state.extend(self.last_movement_direction)
        
        # NEW: Pickup urgency signals (3 values): [health_urgency, machinegun_urgency, ammo_urgency]
        urgency = self._compute_pickup_urgency()
        state.extend([
            urgency['health'],
            urgency['machinegun'],
            urgency['ammo']
        ])
        
        # NEW: Count of in-bounds vs total zombies (2 values)
        in_bounds_count = len(self._get_in_bounds_zombies())
        total_count = len(self.zombies)
        state.extend([
            min(in_bounds_count / 15.0, 1.0),  # Normalized in-bounds count
            min(total_count / 20.0, 1.0)  # Normalized total count
        ])
        
        # Verify state dimension (was 111, now 111 + 1 (zombie in_bounds flag) + 3 (urgency) + 2 (counts) - 1 (removed from zombie count) = 116)
        # Actually: 6 + 3 + (13*7) + 18 + 1 + 1 + 4 + 3 + 2 = 6+3+91+18+1+1+4+3+2 = 129
        # Let me recalculate: zombies now have 7 values each (added in_bounds flag)
        # 6 + 3 + (13*7=91) + 18 + 1 + 1 + 4 + 3 + 2 = 129
        assert len(state) == 129, f"State dimension mismatch: expected 129, got {len(state)}"
        
        return np.array(state, dtype=np.float32)
    
    def _create_player(self) -> Dict:
        """Create a new player with full weapon support."""
        return {
            'x': self.ARENA_X_OFFSET + self.ARENA_WIDTH // 2,  # Center of arena
            'y': self.ARENA_Y_OFFSET + self.ARENA_HEIGHT // 2,  # Center of arena
            'angle': 0,
            'health': 100,
            'score': 0,
            'shoot_cooldown': 0,
            'weapon': 'pistol',  # Current weapon
            'has_machinegun': False,  # Must pick up to use
            'machinegun_ammo': 0
        }
    
    def _spawn_zombie(self) -> Dict:
        """
        Spawn a new zombie uniformly along the full perimeter of the playable arena.
        
        Zombies spawn directly on the arena perimeter boundary (inside the arena).
        Spawn positions are uniformly distributed along all four sides.
        """
        # Randomly select which side of the perimeter to spawn on
        side = random.randint(0, 3)
        
        if side == 0:  # Top side - spawn along full width
            x = random.uniform(self.ARENA_X_OFFSET, self.ARENA_X_OFFSET + self.ARENA_WIDTH)
            y = self.ARENA_Y_OFFSET  # Spawn on top edge of arena
        elif side == 1:  # Right side - spawn along full height
            x = self.ARENA_X_OFFSET + self.ARENA_WIDTH  # Spawn on right edge of arena
            y = random.uniform(self.ARENA_Y_OFFSET, self.ARENA_Y_OFFSET + self.ARENA_HEIGHT)
        elif side == 2:  # Bottom side - spawn along full width
            x = random.uniform(self.ARENA_X_OFFSET, self.ARENA_X_OFFSET + self.ARENA_WIDTH)
            y = self.ARENA_Y_OFFSET + self.ARENA_HEIGHT  # Spawn on bottom edge of arena
        else:  # Left side - spawn along full height
            x = self.ARENA_X_OFFSET  # Spawn on left edge of arena
            y = random.uniform(self.ARENA_Y_OFFSET, self.ARENA_Y_OFFSET + self.ARENA_HEIGHT)
        
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
        
        # Simple rendering - fill expanded screen
        self.screen.fill((96, 96, 96))
        
        # Draw arena background (darker to distinguish playable area)
        arena_rect = pygame.Rect(
            self.ARENA_X_OFFSET,
            self.ARENA_Y_OFFSET,
            self.ARENA_WIDTH,
            self.ARENA_HEIGHT
        )
        pygame.draw.rect(self.screen, (80, 80, 80), arena_rect)
        
        # Draw glass borders around the arena (only on left and right sides as requested)
        # Create a surface for transparency
        glass_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Left glass border (full height of arena)
        left_border = pygame.Rect(
            self.ARENA_X_OFFSET - self.GLASS_BORDER_THICKNESS,
            self.ARENA_Y_OFFSET,
            self.GLASS_BORDER_THICKNESS,
            self.ARENA_HEIGHT
        )
        pygame.draw.rect(glass_surface, self.GLASS_COLOR, left_border)
        
        # Right glass border (full height of arena)
        right_border = pygame.Rect(
            self.ARENA_X_OFFSET + self.ARENA_WIDTH,
            self.ARENA_Y_OFFSET,
            self.GLASS_BORDER_THICKNESS,
            self.ARENA_HEIGHT
        )
        pygame.draw.rect(glass_surface, self.GLASS_COLOR, right_border)
        
        # Blit the glass surface onto the screen
        self.screen.blit(glass_surface, (0, 0))
        
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
