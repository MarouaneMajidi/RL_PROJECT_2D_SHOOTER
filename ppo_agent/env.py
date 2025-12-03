"""
Gym-style Environment Wrapper for Zombie Shooter

Wraps the Pygame game into a gym-like interface for RL training.
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
    
    Action Space (Discrete):
        0: Move Up
        1: Move Down
        2: Move Left
        3: Move Right
        4: Shoot
        5: Idle (no action)
    
    State Space:
        - Player: [x, y, vel_x, vel_y, health, shoot_cooldown, angle, zombie_count]
        - Per zombie (up to max_zombies): [x, y, health, type, distance, angle_to_player]
    """
    
    # Game constants (imported from main.py)
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    PLAYER_SPEED = 5
    ZOMBIE_SPEED = 2
    BULLET_SPEED = 10
    SPAWN_RATE = 60
    
    ZOMBIE_TYPE2_SPAWN_PROBABILITY = 0.2
    HEALTH_PICKUP_DROP_PROBABILITY = 0.40
    MACHINEGUN_PICKUP_DROP_PROBABILITY = 0.15
    
    PISTOL_COOLDOWN = 20
    MACHINEGUN_COOLDOWN = 5
    PISTOL_DAMAGE = 20
    MACHINEGUN_DAMAGE = 10
    
    ZOMBIE_NORMAL_HEALTH = 30
    ZOMBIE_NORMAL_DAMAGE = 10
    ZOMBIE_STRONG_HEALTH = 50
    ZOMBIE_STRONG_DAMAGE = 20
    ZOMBIE_ATTACK_COOLDOWN = 9
    
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
        self.total_reward = 0
        self.wave = 1  # Track current wave
        
        # Tracking for reward shaping
        self.last_player_pos = None
        self.idle_frames = 0
        self.last_health = 100
        self.previous_min_zombie_dist = float('inf')
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Reset game objects
        self.player = self._create_player()
        self.zombies = []
        self.bullets = []
        self.pickups = []
        
        # Reset tracking
        self.episode_steps = 0
        self.spawn_timer = 0
        self.spawn_rate = self.SPAWN_RATE  # Reset spawn rate
        self.zombies_killed = 0
        self.total_reward = 0
        self.wave = 1  # Reset wave
        self.idle_frames = 0
        self.last_player_pos = (self.player['x'], self.player['y'])
        self.last_health = self.player['health']
        self.previous_min_zombie_dist = float('inf')
        
        # Spawn initial zombies
        for _ in range(3):
            self.zombies.append(self._spawn_zombie())
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-5)
        
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.episode_steps += 1
        reward = 0.0
        info = {}
        
        # Apply action
        self._apply_action(action)
        
        # Update game state
        self._update_bullets()
        self._update_zombies()
        pickup_reward = self._update_pickups()
        
        # Add pickup collection reward
        reward += pickup_reward
        
        # Spawn new zombies with wave mechanics
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.zombies.append(self._spawn_zombie())
            self.spawn_timer = 0
            
            # Increase difficulty every 10 zombies killed (wave progression)
            if self.zombies_killed > 0 and self.zombies_killed % 10 == 0:
                if self.spawn_rate > 10:  # Don't go below 10 frames
                    old_rate = self.spawn_rate
                    self.spawn_rate = max(10, self.spawn_rate - 5)
                    if old_rate != self.spawn_rate:
                        self.wave += 1
        
        # Compute rewards
        reward += self.config.reward_survival_per_step
        
        # Reward for killing zombies (tracked in bullet collision)
        # This is handled by checking zombies_killed delta
        
        # Penalty for damage taken
        if self.player['health'] < self.last_health:
            damage_taken = self.last_health - self.player['health']
            reward += self.config.reward_damage_taken * (damage_taken / 10)
        self.last_health = self.player['health']
        
        # Penalty for staying idle
        current_pos = (self.player['x'], self.player['y'])
        if current_pos == self.last_player_pos:
            self.idle_frames += 1
            if self.idle_frames > 30:  # If idle for more than 0.5 seconds
                reward += self.config.reward_idle_penalty
        else:
            self.idle_frames = 0
        self.last_player_pos = current_pos
        
        # Small reward for getting closer to zombies (encourages engagement)
        if len(self.zombies) > 0:
            min_dist = min([self._distance(self.player['x'], self.player['y'], z['x'], z['y']) 
                           for z in self.zombies])
            if min_dist < self.previous_min_zombie_dist:
                reward += self.config.reward_distance_to_zombie
            self.previous_min_zombie_dist = min_dist
        
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
        
        return state, reward, done, info
    
    def _apply_action(self, action: int):
        """Apply the given action to the player."""
        # Movement actions
        if action == 0:  # Move Up
            self.player['y'] -= self.PLAYER_SPEED
        elif action == 1:  # Move Down
            self.player['y'] += self.PLAYER_SPEED
        elif action == 2:  # Move Left
            self.player['x'] -= self.PLAYER_SPEED
        elif action == 3:  # Move Right
            self.player['x'] += self.PLAYER_SPEED
        elif action == 4:  # Shoot
            self._player_shoot()
        # action == 5: Idle (do nothing)
        
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
    
    def _player_shoot(self):
        """Handle player shooting."""
        if self.player['shoot_cooldown'] == 0:
            angle_rad = math.radians(self.player['angle'])
            self.bullets.append({
                'x': self.player['x'],
                'y': self.player['y'],
                'angle': angle_rad,
                'damage': self.PISTOL_DAMAGE
            })
            self.player['shoot_cooldown'] = self.PISTOL_COOLDOWN
    
    def _update_bullets(self):
        """Update bullet positions and check collisions."""
        bullets_to_remove = []
        
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
                    if zombie['health'] <= 0:
                        zombies_to_remove.append(zombie)
                        self.player['score'] += 10
                        self.zombies_killed += 1
                        # Add kill reward immediately
                        self.total_reward += self.config.reward_zombie_kill
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
    
    def _update_pickups(self):
        """Update pickups and check collisions."""
        pickups_to_remove = []
        pickup_reward = 0.0
        
        for pickup in self.pickups:
            pickup['lifetime'] -= 1
            if pickup['lifetime'] <= 0:
                pickups_to_remove.append(pickup)
            elif self._distance(pickup['x'], pickup['y'], self.player['x'], self.player['y']) < 25:
                if pickup['type'] == 'health':
                    self.player['health'] = min(100, self.player['health'] + 25)
                    pickup_reward += self.config.reward_health_pickup
                elif pickup['type'] == 'machinegun':
                    # Machine gun pickup (not fully implemented in env but we reward it)
                    pickup_reward += self.config.reward_machinegun_pickup
                pickups_to_remove.append(pickup)
        
        for pickup in pickups_to_remove:
            if pickup in self.pickups:
                self.pickups.remove(pickup)
        
        return pickup_reward
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            State vector as numpy array
        """
        state = []
        
        # Player state: [x, y, vel_x, vel_y, health, shoot_cooldown, angle, zombie_count]
        state.extend([
            self.player['x'] / self.SCREEN_WIDTH,  # Normalize
            self.player['y'] / self.SCREEN_HEIGHT,
            0.0,  # vel_x (not tracked in simplified version)
            0.0,  # vel_y
            self.player['health'] / 100.0,
            self.player['shoot_cooldown'] / self.PISTOL_COOLDOWN,
            self.player['angle'] / 360.0,
            min(len(self.zombies), self.max_zombies) / self.max_zombies
        ])
        
        # Zombie states (sort by distance, take closest max_zombies)
        zombies_sorted = sorted(self.zombies, 
                               key=lambda z: self._distance(self.player['x'], self.player['y'], z['x'], z['y']))
        
        for i in range(self.max_zombies):
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
        
        return np.array(state, dtype=np.float32)
    
    def _create_player(self) -> Dict:
        """Create a new player."""
        return {
            'x': self.SCREEN_WIDTH // 2,
            'y': self.SCREEN_HEIGHT // 2,
            'angle': 0,
            'health': 100,
            'score': 0,
            'shoot_cooldown': 0
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
                'lifetime': 600
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
        pygame.draw.circle(self.screen, (0, 255, 0), 
                          (int(self.player['x']), int(self.player['y'])), 15)
        
        # Draw zombies
        for zombie in self.zombies:
            color = (255, 0, 0) if zombie['type'] == 'normal' else (150, 0, 0)
            pygame.draw.circle(self.screen, color, 
                             (int(zombie['x']), int(zombie['y'])), 12)
        
        # Draw bullets
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 0), 
                             (int(bullet['x']), int(bullet['y'])), 4)
        
        # Draw UI
        font = pygame.font.SysFont(None, 24)
        health_text = font.render(f"Health: {self.player['health']}", True, (255, 255, 255))
        score_text = font.render(f"Score: {self.player['score']}", True, (255, 255, 255))
        kills_text = font.render(f"Kills: {self.zombies_killed}", True, (255, 255, 255))
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(score_text, (10, 35))
        self.screen.blit(kills_text, (10, 60))
        
        pygame.display.flip()
        self.clock.tick(self.FPS)
    
    def close(self):
        """Close the environment."""
        pygame.quit()
