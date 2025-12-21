"""
Top-Down Zombie Shooter

A 2D top-down shooter game with zombie enemies.
Supports two modes:
1. Manual mode - Player controls with keyboard and mouse
2. Agent mode - PPO reinforcement learning agent plays the game

FIXED ISSUES:
1. Wave reset bug - wave, spawn_rate, and all tracking now properly reset
2. Machine gun support in agent mode - agent can switch weapons
3. Added comprehensive resetGame() method for clean restarts
"""

import os
import pygame
import sys
import math
import random
import argparse
from pygame.locals import *

# Asset constants
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Import PPO agent (optional, only if agent_mode is used)
try:
    from agents.ppo_agent import PPOAgent, PPOConfig
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

# Initialize pygame
pygame.init()

# Game constants
# Arena (playable area) - stays exactly the same
ARENA_WIDTH = 800
ARENA_HEIGHT = 600

# Expanded screen size (visible area, larger than arena)
SCREEN_WIDTH = 850  # Expanded to show area around arena
SCREEN_HEIGHT = 650  # Expanded to show area around arena

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
SPAWN_RATE = 60  # Initial frames between zombie spawns

# Probability and Spawn Settings
ZOMBIE_TYPE2_SPAWN_PROBABILITY = 0.2
HEALTH_PICKUP_DROP_PROBABILITY = 0.40
MACHINEGUN_PICKUP_DROP_PROBABILITY = 0.15

# Pickup Configuration
HEALTH_PICKUP_AMOUNT = 25
MACHINEGUN_PICKUP_AMMO = 50
PICKUP_LIFETIME = 600

# Weapon Configuration
PISTOL_COOLDOWN = 20
MACHINEGUN_COOLDOWN = 5
KNIFE_COOLDOWN = 30
PISTOL_DAMAGE = 20
MACHINEGUN_DAMAGE = 10
KNIFE_DAMAGE = 40
KNIFE_RANGE = 50
KNIFE_LIFETIME = 5

# Zombie Configuration
ZOMBIE_NORMAL_HEALTH = 30
ZOMBIE_NORMAL_DAMAGE = 10
ZOMBIE_STRONG_HEALTH = 50
ZOMBIE_STRONG_DAMAGE = 20
ZOMBIE_ATTACK_COOLDOWN = 9

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
BG_COLOR = (96, 96, 96)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Top-Down Shooter")
clock = pygame.time.Clock()

def load_sprite(filename, scale=1.0):
    """Load a sprite from disk and scale it so the gifs can be reused."""
    path = os.path.join(ASSET_DIR, filename)
    image = pygame.image.load(path).convert_alpha()
    if scale != 1.0:
        new_size = (int(image.get_width() * scale), int(image.get_height() * scale))
        image = pygame.transform.scale(image, new_size)
    return image

# Load the provided gif sprites
player_pistol_img = load_sprite("player pistol.gif", 1.2)
player_machinegun_img = load_sprite("player machinegun.gif", 1.2)
player_knife_img = load_sprite("player knife.gif", 1.2)
zombie_img = load_sprite("zombie.gif", 1.2)
zombie2_img = load_sprite("zombie 2.gif", 1.2)
background_img = pygame.image.load(os.path.join(ASSET_DIR, "background.png")).convert()
# Scale background to arena size (playable area)
background_img = pygame.transform.scale(background_img, (ARENA_WIDTH, ARENA_HEIGHT))


class Player:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset player to initial state."""
        self.x = ARENA_X_OFFSET + ARENA_WIDTH // 2  # Center of arena
        self.y = ARENA_Y_OFFSET + ARENA_HEIGHT // 2  # Center of arena
        self.angle = 0
        self.weapon = "pistol"  # pistol, machinegun, knife
        self.health = 100
        self.score = 0
        self.shoot_cooldown = 0
        self.knife_cooldown = 0
        self.speed = PLAYER_SPEED
        self.has_machinegun = False
        self.machinegun_ammo = 0
        
    def update(self, keys, mouse_pos, mouse_buttons):
        # Calculate angle to mouse position
        dx = mouse_pos[0] - self.x
        dy = mouse_pos[1] - self.y
        self.angle = math.degrees(math.atan2(-dy, dx)) % 360
        
        # Movement
        if keys[K_w] or keys[K_UP]:
            self.y -= self.speed
        if keys[K_s] or keys[K_DOWN]:
            self.y += self.speed
        if keys[K_a] or keys[K_LEFT]:
            self.x -= self.speed
        if keys[K_d] or keys[K_RIGHT]:
            self.x += self.speed
            
        # Keep player within arena bounds (glass borders prevent movement outside)
        self.x = max(ARENA_X_OFFSET, min(ARENA_X_OFFSET + ARENA_WIDTH, self.x))
        self.y = max(ARENA_Y_OFFSET, min(ARENA_Y_OFFSET + ARENA_HEIGHT, self.y))
        
        # Weapon switching
        if keys[K_1]:
            self.weapon = "pistol"
        if keys[K_2] and self.has_machinegun:
            self.weapon = "machinegun"
        if keys[K_3]:
            self.weapon = "knife"
            
        # Shooting
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            
        if self.knife_cooldown > 0:
            self.knife_cooldown -= 1
            
        if mouse_buttons[0] and self.shoot_cooldown == 0 and self.weapon != "knife":
            if self.weapon == "pistol":
                self.shoot_cooldown = PISTOL_COOLDOWN
                return Bullet(self.x, self.y, self.angle, "pistol")
            elif self.weapon == "machinegun" and self.has_machinegun and self.machinegun_ammo > 0:
                self.shoot_cooldown = MACHINEGUN_COOLDOWN
                self.machinegun_ammo -= 1
                # Auto-switch to pistol when out of ammo
                if self.machinegun_ammo <= 0:
                    self.has_machinegun = False
                    self.weapon = "pistol"
                return Bullet(self.x, self.y, self.angle, "machinegun")
                
        # Knife attack
        if mouse_buttons[0] and self.knife_cooldown == 0 and self.weapon == "knife":
            self.knife_cooldown = KNIFE_COOLDOWN
            return KnifeAttack(self.x, self.y, self.angle)
            
        return None
        
    def draw(self, screen):
        # Draw the appropriate player image based on weapon
        if self.weapon == "pistol":
            img = player_pistol_img
        elif self.weapon == "machinegun" and self.has_machinegun:
            img = player_machinegun_img
        else:
            img = player_knife_img if self.weapon == "knife" else player_pistol_img
            
        # Rotate image to face mouse
        rotated_img = pygame.transform.rotate(img, self.angle)
        rect = rotated_img.get_rect(center=(self.x, self.y))
        screen.blit(rotated_img, rect)
        
        # Draw health bar
        pygame.draw.rect(screen, RED, (self.x - 25, self.y - 40, 50, 5))
        pygame.draw.rect(screen, GREEN, (self.x - 25, self.y - 40, 50 * (self.health / 100), 5))
        
    def add_health(self, amount):
        self.health = min(100, self.health + amount)
        
    def pickup_machinegun(self, ammo=MACHINEGUN_PICKUP_AMMO):
        self.has_machinegun = True
        self.machinegun_ammo += ammo
        
    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0


class Zombie:
    def __init__(self, x, y, zombie_type="normal"):
        self.x = x
        self.y = y
        self.zombie_type = zombie_type
        self.speed = ZOMBIE_SPEED
        self.health = ZOMBIE_NORMAL_HEALTH if zombie_type == "normal" else ZOMBIE_STRONG_HEALTH
        self.damage = ZOMBIE_NORMAL_DAMAGE if zombie_type == "normal" else ZOMBIE_STRONG_DAMAGE
        self.base_image = zombie_img if zombie_type == "normal" else zombie2_img
        self.angle = 0
        self.attack_cooldown = 0
        
    def update(self, player_x, player_y):
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            
        dx = player_x - self.x
        dy = player_y - self.y
        dist = max(0.1, math.hypot(dx, dy))
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        self.angle = math.degrees(math.atan2(-dy, dx)) % 360
        
    def can_attack(self):
        return self.attack_cooldown == 0
        
    def attack(self):
        self.attack_cooldown = ZOMBIE_ATTACK_COOLDOWN
        return self.damage
        
    def draw(self, screen):
        rotated = pygame.transform.rotate(self.base_image, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)
        
        health_width = 30
        pygame.draw.rect(screen, RED, (self.x - health_width//2, self.y - 30, health_width, 5))
        max_health = ZOMBIE_NORMAL_HEALTH if self.zombie_type == "normal" else ZOMBIE_STRONG_HEALTH
        pygame.draw.rect(screen, GREEN, (self.x - health_width//2, self.y - 30, health_width * (self.health / max_health), 5))
        
    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0
        
    def check_collision_with_player(self, player_x, player_y):
        distance = math.hypot(player_x - self.x, player_y - self.y)
        return distance < 30


class Bullet:
    def __init__(self, x, y, angle, weapon_type):
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.weapon_type = weapon_type
        self.speed = BULLET_SPEED
        self.damage = PISTOL_DAMAGE if weapon_type == "pistol" else MACHINEGUN_DAMAGE
        
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y -= math.sin(self.angle) * self.speed
        # Check if bullet is off the expanded screen
        return (self.x < 0 or self.x > SCREEN_WIDTH or 
                self.y < 0 or self.y > SCREEN_HEIGHT)
        
    def draw(self, screen):
        color = YELLOW if self.weapon_type == "pistol" else (255, 165, 0)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 4)
        
    def check_collision(self, zombie):
        distance = math.hypot(self.x - zombie.x, self.y - zombie.y)
        return distance < 25


class KnifeAttack:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.range = KNIFE_RANGE
        self.damage = KNIFE_DAMAGE
        self.lifetime = KNIFE_LIFETIME
        
    def update(self):
        self.lifetime -= 1
        return self.lifetime <= 0
        
    def draw(self, screen):
        start_angle = self.angle - math.radians(45)
        end_angle = self.angle + math.radians(45)
        pygame.draw.arc(screen, WHITE, 
                       (self.x - self.range, self.y - self.range, 
                        self.range * 2, self.range * 2),
                       start_angle, end_angle, 3)
        
    def check_collision(self, zombie):
        distance = math.hypot(self.x - zombie.x, self.y - zombie.y)
        if distance > self.range:
            return False
        dx = zombie.x - self.x
        dy = zombie.y - self.y
        zombie_angle = math.atan2(-dy, dx) % (2 * math.pi)
        attack_angle = self.angle % (2 * math.pi)
        angle_diff = abs(zombie_angle - attack_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        return angle_diff <= math.radians(45)


class Pickup:
    def __init__(self, x, y, pickup_type):
        self.x = x
        self.y = y
        self.pickup_type = pickup_type
        self.lifetime = PICKUP_LIFETIME
        
    def update(self):
        self.lifetime -= 1
        return self.lifetime <= 0
        
    def draw(self, screen):
        if self.pickup_type == "health":
            pygame.draw.rect(screen, RED, (self.x - 15, self.y - 3, 30, 6))
            pygame.draw.rect(screen, RED, (self.x - 3, self.y - 15, 6, 30))
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), 18, 2)
        elif self.pickup_type == "machinegun":
            pygame.draw.rect(screen, (100, 100, 100), (self.x - 12, self.y - 4, 24, 8))
            pygame.draw.rect(screen, (150, 150, 150), (self.x - 8, self.y - 2, 16, 4))
            pygame.draw.circle(screen, YELLOW, (int(self.x + 10), int(self.y)), 3)
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), 18, 2)
        elif self.pickup_type == "ammo":
            # Ammo pickup: yellow box with bullets
            pygame.draw.rect(screen, YELLOW, (self.x - 10, self.y - 5, 20, 10))
            pygame.draw.rect(screen, (200, 200, 0), (self.x - 8, self.y - 3, 16, 6))
            # Draw bullet indicators
            for i in range(3):
                pygame.draw.circle(screen, (100, 100, 0), (int(self.x - 5 + i * 5), int(self.y)), 2)
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), 18, 2)
            
    def check_collision(self, player_x, player_y):
        distance = math.hypot(player_x - self.x, player_y - self.y)
        return distance < 25


class GameState:
    """
    Manages all game state in one place.
    Provides a clean resetGame() method that ensures everything is properly reset.
    
    FIXES THE WAVE RESET BUG:
    - All state variables are centralized here
    - resetGame() resets EVERYTHING including wave and spawn_rate
    - No more desync between UI wave counter and internal logic
    """
    
    def __init__(self):
        self.player = None
        self.zombies = []
        self.bullets = []
        self.knife_attacks = []
        self.pickups = []
        self.zombies_killed = 0
        self.wave = 1
        self.spawn_timer = 0
        self.spawn_rate = SPAWN_RATE
        self.game_over = False
        
    def resetGame(self):
        """
        COMPLETE GAME RESET - Fixes the wave reset bug.
        
        This method ensures ALL game state is reset to initial values:
        1. Player state (position, health, weapons)
        2. Enemy lists (zombies, bullets, etc.)
        3. Wave counter and spawn rate
        4. All timers and tracking variables
        
        Call this when:
        - Starting a new game
        - Player dies and presses R to retry
        - Resetting the environment in RL training
        """
        # Reset player
        if self.player is None:
            self.player = Player()
        else:
            self.player.reset()
        
        # Clear all game objects
        self.zombies = []
        self.bullets = []
        self.knife_attacks = []
        self.pickups = []
        
        # CRITICAL: Reset wave and spawn rate to initial values
        self.wave = 1
        self.spawn_rate = SPAWN_RATE  # Reset to initial spawn rate!
        
        # Reset tracking
        self.zombies_killed = 0
        self.spawn_timer = 0
        self.game_over = False
        
        return self


def spawn_zombie():
    """
    Spawn a new zombie uniformly along the full perimeter of the playable arena.
    
    Zombies spawn directly on the arena perimeter boundary (inside the arena).
    Spawn positions are uniformly distributed along all four sides.
    """
    # Randomly select which side of the perimeter to spawn on
    side = random.randint(0, 3)
    
    if side == 0:  # Top side - spawn along full width
        x = random.uniform(ARENA_X_OFFSET, ARENA_X_OFFSET + ARENA_WIDTH)
        y = ARENA_Y_OFFSET  # Spawn on top edge of arena
    elif side == 1:  # Right side - spawn along full height
        x = ARENA_X_OFFSET + ARENA_WIDTH  # Spawn on right edge of arena
        y = random.uniform(ARENA_Y_OFFSET, ARENA_Y_OFFSET + ARENA_HEIGHT)
    elif side == 2:  # Bottom side - spawn along full width
        x = random.uniform(ARENA_X_OFFSET, ARENA_X_OFFSET + ARENA_WIDTH)
        y = ARENA_Y_OFFSET + ARENA_HEIGHT  # Spawn on bottom edge of arena
    else:  # Left side - spawn along full height
        x = ARENA_X_OFFSET  # Spawn on left edge of arena
        y = random.uniform(ARENA_Y_OFFSET, ARENA_Y_OFFSET + ARENA_HEIGHT)
    zombie_type = "strong" if random.random() < ZOMBIE_TYPE2_SPAWN_PROBABILITY else "normal"
    return Zombie(x, y, zombie_type)


def draw_ui(screen, player, wave, zombies_killed):
    font = pygame.font.SysFont(None, 36)
    
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    
    wave_text = font.render(f"Wave: {wave}", True, WHITE)
    screen.blit(wave_text, (10, 50))
    
    kills_text = font.render(f"Zombies Killed: {zombies_killed}", True, WHITE)
    screen.blit(kills_text, (10, 90))
    
    weapon_text = font.render(f"Weapon: {player.weapon.capitalize()}", True, WHITE)
    screen.blit(weapon_text, (SCREEN_WIDTH - 200, 10))
    
    if player.has_machinegun:
        keys_text = font.render("1:Pistol  2:Machine Gun  3:Knife", True, WHITE)
    else:
        keys_text = font.render("1:Pistol  [Locked]  3:Knife", True, WHITE)
    screen.blit(keys_text, (SCREEN_WIDTH - 300, 50))
    
    health_text = font.render(f"Health: {player.health}", True, WHITE)
    screen.blit(health_text, (SCREEN_WIDTH - 150, 90))
    
    if player.has_machinegun:
        ammo_text = font.render(f"MG Ammo: {player.machinegun_ammo}", True, YELLOW)
        screen.blit(ammo_text, (SCREEN_WIDTH - 200, 130))


def draw_game_over(screen, score, zombies_killed, wave):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))
    
    font_large = pygame.font.SysFont(None, 72)
    font_medium = pygame.font.SysFont(None, 48)
    
    game_over_text = font_large.render("GAME OVER", True, RED)
    screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 100))
    
    score_text = font_medium.render(f"Final Score: {score}", True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH//2 - score_text.get_width()//2, SCREEN_HEIGHT//2))
    
    kills_text = font_medium.render(f"Zombies Killed: {zombies_killed}", True, WHITE)
    screen.blit(kills_text, (SCREEN_WIDTH//2 - kills_text.get_width()//2, SCREEN_HEIGHT//2 + 50))
    
    wave_text = font_medium.render(f"Wave Reached: {wave}", True, WHITE)
    screen.blit(wave_text, (SCREEN_WIDTH//2 - wave_text.get_width()//2, SCREEN_HEIGHT//2 + 100))
    
    restart_text = font_medium.render("Press R to Restart", True, GREEN)
    screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2 + 180))
    
    quit_text = font_medium.render("Press Q to Quit", True, RED)
    screen.blit(quit_text, (SCREEN_WIDTH//2 - quit_text.get_width()//2, SCREEN_HEIGHT//2 + 230))


def main_manual_mode():
    """
    Manual mode - player controls with keyboard and mouse.
    
    FIXED: Uses GameState class for proper wave reset on player death.
    """
    # Initialize game state using the new GameState class
    game = GameState()
    game.resetGame()
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if game.game_over and event.key == K_r:
                    # FIXED: Use resetGame() for complete reset including wave
                    game.resetGame()
                if game.game_over and event.key == K_q:
                    pygame.quit()
                    sys.exit()
        
        if not game.game_over:
            # Get input
            keys = pygame.key.get_pressed()
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            # Update player
            new_bullet = game.player.update(keys, mouse_pos, mouse_buttons)
            if new_bullet:
                if isinstance(new_bullet, Bullet):
                    game.bullets.append(new_bullet)
                else:
                    game.knife_attacks.append(new_bullet)
            
            # Update pickups and check collisions
            pickups_to_remove = []
            for pickup in game.pickups:
                if pickup.update():
                    pickups_to_remove.append(pickup)
                elif pickup.check_collision(game.player.x, game.player.y):
                    if pickup.pickup_type == "health":
                        game.player.add_health(HEALTH_PICKUP_AMOUNT)
                    elif pickup.pickup_type == "machinegun":
                        game.player.pickup_machinegun(MACHINEGUN_PICKUP_AMMO)
                    pickups_to_remove.append(pickup)
            
            for pickup in pickups_to_remove:
                if pickup in game.pickups:
                    game.pickups.remove(pickup)
            
            # Update bullets
            bullets_to_remove = []
            for bullet in game.bullets:
                if bullet.update():
                    bullets_to_remove.append(bullet)
                else:
                    zombies_to_remove = []
                    for zombie in game.zombies:
                        if bullet.check_collision(zombie):
                            if zombie.take_damage(bullet.damage):
                                zombies_to_remove.append(zombie)
                                game.player.score += 10 if zombie.zombie_type == "normal" else 20
                                game.zombies_killed += 1
                                drop_chance = random.random()
                                if drop_chance < HEALTH_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "health"))
                                elif drop_chance < HEALTH_PICKUP_DROP_PROBABILITY + MACHINEGUN_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "machinegun"))
                            bullets_to_remove.append(bullet)
                            break
                    for zombie in zombies_to_remove:
                        game.zombies.remove(zombie)
            
            for bullet in bullets_to_remove:
                if bullet in game.bullets:
                    game.bullets.remove(bullet)
            
            # Update knife attacks
            knife_attacks_to_remove = []
            for knife in game.knife_attacks:
                if knife.update():
                    knife_attacks_to_remove.append(knife)
                else:
                    zombies_to_remove = []
                    for zombie in game.zombies:
                        if knife.check_collision(zombie):
                            if zombie.take_damage(knife.damage):
                                zombies_to_remove.append(zombie)
                                game.player.score += 10 if zombie.zombie_type == "normal" else 20
                                game.zombies_killed += 1
                                drop_chance = random.random()
                                if drop_chance < HEALTH_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "health"))
                                elif drop_chance < HEALTH_PICKUP_DROP_PROBABILITY + MACHINEGUN_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "machinegun"))
                    for zombie in zombies_to_remove:
                        game.zombies.remove(zombie)
            
            for knife in knife_attacks_to_remove:
                if knife in game.knife_attacks:
                    game.knife_attacks.remove(knife)
            
            # Spawn zombies with wave progression
            game.spawn_timer += 1
            if game.spawn_timer >= game.spawn_rate:
                game.zombies.append(spawn_zombie())
                game.spawn_timer = 0
                
                # Increase spawn rate every 10 zombies killed
                if game.zombies_killed > 0 and game.zombies_killed % 10 == 0:
                    expected_wave = (game.zombies_killed // 10) + 1
                    if expected_wave > game.wave:
                        game.wave = expected_wave
                        game.spawn_rate = max(10, game.spawn_rate - 5)
            
            # Update zombies
            for zombie in game.zombies:
                zombie.update(game.player.x, game.player.y)
                if zombie.check_collision_with_player(game.player.x, game.player.y):
                    if zombie.can_attack():
                        damage = zombie.attack()
                        if game.player.take_damage(damage):
                            game.game_over = True
        
        # Draw everything
        # Fill expanded screen background
        screen.fill(BG_COLOR)
        
        # Draw arena background (playable area)
        arena_rect = pygame.Rect(
            ARENA_X_OFFSET,
            ARENA_Y_OFFSET,
            ARENA_WIDTH,
            ARENA_HEIGHT
        )
        if background_img:
            screen.blit(background_img, (ARENA_X_OFFSET, ARENA_Y_OFFSET))
        else:
            pygame.draw.rect(screen, (80, 80, 80), arena_rect)
        
        # Draw glass borders around the arena (only on left and right sides as requested)
        glass_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Left glass border (full height of arena)
        left_border = pygame.Rect(
            ARENA_X_OFFSET - GLASS_BORDER_THICKNESS,
            ARENA_Y_OFFSET,
            GLASS_BORDER_THICKNESS,
            ARENA_HEIGHT
        )
        pygame.draw.rect(glass_surface, GLASS_COLOR, left_border)
        
        # Right glass border (full height of arena)
        right_border = pygame.Rect(
            ARENA_X_OFFSET + ARENA_WIDTH,
            ARENA_Y_OFFSET,
            GLASS_BORDER_THICKNESS,
            ARENA_HEIGHT
        )
        pygame.draw.rect(glass_surface, GLASS_COLOR, right_border)
        
        # Blit the glass surface onto the screen
        screen.blit(glass_surface, (0, 0))
        
        game.player.draw(screen)
        for zombie in game.zombies:
            zombie.draw(screen)
        for bullet in game.bullets:
            bullet.draw(screen)
        for knife in game.knife_attacks:
            knife.draw(screen)
        for pickup in game.pickups:
            pickup.draw(screen)
        
        draw_ui(screen, game.player, game.wave, game.zombies_killed)
        
        if game.game_over:
            draw_game_over(screen, game.player.score, game.zombies_killed, game.wave)
        
        pygame.display.flip()
        clock.tick(FPS)


def main_agent_mode(model_path: str):
    """
    Agent mode - PPO agent controls the player using FULL game with assets.
    
    FIXED:
    1. Uses GameState class for proper wave reset
    2. Machine gun support with weapon switching actions
    3. Extended action space (8 actions)
    
    Args:
        model_path: Path to trained PPO model checkpoint
    """
    if not PPO_AVAILABLE:
        print("Error: PPO agent not available. Please install PyTorch and train a model first.")
        return
    
    print(f"Loading PPO agent from: {model_path}")
    from ppo_agent import PPOConfig, PPOAgent
    config = PPOConfig()
    agent = PPOAgent(config)
    
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using: python train_ppo.py")
        return
    
    print("\nAgent is now playing with FULL game assets!")
    print("Action space: Multi-discrete [Movement (5) + Shoot (2)] - allows simultaneous actions!")
    print("Press ESC to quit, R to restart episode")
    print("="*50)
    
    import numpy as np
    
    # Initialize game state using GameState class
    game = GameState()
    game.resetGame()
    
    episode_reward = 0
    episode_steps = 0
    episodes = 0
    
    def is_zombie_in_bounds(zombie):
        """Check if zombie is within playable arena area (inside glass borders)."""
        return (ARENA_X_OFFSET <= zombie.x <= ARENA_X_OFFSET + ARENA_WIDTH and 
                ARENA_Y_OFFSET <= zombie.y <= ARENA_Y_OFFSET + ARENA_HEIGHT)
    
    def get_in_bounds_zombies():
        """Get only zombies within playable area."""
        return [z for z in game.zombies if is_zombie_in_bounds(z)]
    
    def compute_pickup_urgency():
        """Compute urgency scores for each pickup type."""
        urgency = {
            'health': 1.0 - (game.player.health / 100.0),  # Higher when health is low
            'machinegun': 1.0 if not game.player.has_machinegun else 0.0,  # High if don't have it
            'ammo': (1.0 - (game.player.machinegun_ammo / 100.0)) if game.player.has_machinegun else 0.0
        }
        return urgency
    
    def get_agent_state():
        """Extract state for the agent matching the new 129-dim env.py format with enhanced features."""
        state = []
        
        # Player state (6 values): [x, y, health, shoot_cooldown, angle, zombie_count]
        # Normalize player position relative to arena (not screen)
        in_bounds_zombies = get_in_bounds_zombies()
        state.extend([
            (game.player.x - ARENA_X_OFFSET) / ARENA_WIDTH,  # Normalize to arena
            (game.player.y - ARENA_Y_OFFSET) / ARENA_HEIGHT,  # Normalize to arena
            game.player.health / 100.0,
            game.player.shoot_cooldown / PISTOL_COOLDOWN,
            game.player.angle / 360.0,
            min(len(in_bounds_zombies), 13) / 13.0  # Use in-bounds count
        ])
        
        # Weapon state (3 values): [has_machinegun, machinegun_ammo_normalized, current_weapon_is_mg]
        state.extend([
            1.0 if game.player.has_machinegun else 0.0,
            min(game.player.machinegun_ammo / 100.0, 1.0),
            1.0 if game.player.weapon == "machinegun" else 0.0
        ])
        
        # Zombie states (prioritize in-bounds, sort by distance, take closest 13)
        in_bounds_zombies = get_in_bounds_zombies()
        out_of_bounds_zombies = [z for z in game.zombies if not is_zombie_in_bounds(z)]
        
        # Prioritize in-bounds zombies
        zombies_sorted = (sorted(in_bounds_zombies, 
                                key=lambda z: math.hypot(game.player.x - z.x, game.player.y - z.y)) +
                         sorted(out_of_bounds_zombies,
                                key=lambda z: math.hypot(game.player.x - z.x, game.player.y - z.y)))
        
        max_zombies_for_state = 13
        for i in range(max_zombies_for_state):
            if i < len(zombies_sorted):
                zombie = zombies_sorted[i]
                distance = math.hypot(game.player.x - zombie.x, game.player.y - zombie.y)
                dx = zombie.x - game.player.x
                dy = zombie.y - game.player.y
                angle_to_zombie = math.degrees(math.atan2(-dy, dx)) % 360
                is_in_bounds = 1.0 if is_zombie_in_bounds(zombie) else 0.0
                
                # Normalize zombie position relative to arena (for consistency with player)
                state.extend([
                    (zombie.x - ARENA_X_OFFSET) / ARENA_WIDTH,
                    (zombie.y - ARENA_Y_OFFSET) / ARENA_HEIGHT,
                    zombie.health / ZOMBIE_STRONG_HEALTH,
                    1.0 if zombie.zombie_type == 'strong' else 0.0,
                    min(distance / 1000.0, 1.0),
                    angle_to_zombie / 360.0,
                    is_in_bounds  # NEW: in_bounds flag
                ])
            else:
                # Padding for empty zombie slots
                state.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        
        # Pickup info (enhanced with directional info): 18 values total
        # Machinegun pickup: [distance, angle, exists, dx_normalized, dy_normalized, in_range]
        nearest_mg = get_nearest_pickup('machinegun')
        if nearest_mg:
            mg_dist = math.hypot(game.player.x - nearest_mg.x, 
                                game.player.y - nearest_mg.y)
            dx = nearest_mg.x - game.player.x
            dy = nearest_mg.y - game.player.y
            mg_angle = math.degrees(math.atan2(-dy, dx)) % 360
            in_range = 1.0 if mg_dist < 300.0 else 0.0  # pickup_detection_range
            state.extend([
                min(mg_dist / 1000.0, 1.0),
                mg_angle / 360.0,
                1.0,  # Exists
                dx / ARENA_WIDTH,  # Normalized dx (relative to arena)
                dy / ARENA_HEIGHT,  # Normalized dy (relative to arena)
                in_range
            ])
        else:
            state.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No machinegun pickup
        
        # Health pack: [distance, angle, exists, dx_normalized, dy_normalized, in_range]
        nearest_health = get_nearest_pickup('health')
        if nearest_health:
            health_dist = math.hypot(game.player.x - nearest_health.x, 
                                    game.player.y - nearest_health.y)
            dx = nearest_health.x - game.player.x
            dy = nearest_health.y - game.player.y
            health_angle = math.degrees(math.atan2(-dy, dx)) % 360
            in_range = 1.0 if health_dist < 300.0 else 0.0
            state.extend([
                min(health_dist / 1000.0, 1.0),
                health_angle / 360.0,
                1.0,  # Exists
                dx / SCREEN_WIDTH,  # Normalized dx
                dy / SCREEN_HEIGHT,  # Normalized dy
                in_range
            ])
        else:
            state.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No health pack
        
        # Ammo pickup: [distance, angle, exists, dx_normalized, dy_normalized, in_range]
        nearest_ammo = get_nearest_pickup('ammo')
        if nearest_ammo:
            ammo_dist = math.hypot(game.player.x - nearest_ammo.x, 
                                   game.player.y - nearest_ammo.y)
            dx = nearest_ammo.x - game.player.x
            dy = nearest_ammo.y - game.player.y
            ammo_angle = math.degrees(math.atan2(-dy, dx)) % 360
            in_range = 1.0 if ammo_dist < 300.0 else 0.0
            state.extend([
                min(ammo_dist / 1000.0, 1.0),
                ammo_angle / 360.0,
                1.0,  # Exists
                dx / ARENA_WIDTH,  # Normalized dx (relative to arena)
                dy / ARENA_HEIGHT,  # Normalized dy (relative to arena)
                in_range
            ])
        else:
            state.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No ammo pickup
        
        # Distance to nearest IN-BOUNDS zombie (1 value)
        in_bounds_zombies = get_in_bounds_zombies()
        if len(in_bounds_zombies) > 0:
            min_dist = min([math.hypot(game.player.x - z.x, game.player.y - z.y) 
                           for z in in_bounds_zombies])
            state.append(min(min_dist / 1000.0, 1.0))  # Normalize
        else:
            state.append(1.0)  # No in-bounds zombies, max distance
        
        # Zombie density in radius (in-bounds only) (1 value)
        radius = 200.0  # Danger radius
        density = sum(1 for z in in_bounds_zombies 
                     if math.hypot(game.player.x - z.x, game.player.y - z.y) < radius)
        state.append(min(density / 10.0, 1.0))  # Normalize (max 10 zombies)
        
        # Movement direction (4 values): [up, down, left, right]
        # For main.py, we'll use zeros (not tracked in main game loop)
        state.extend([0.0, 0.0, 0.0, 0.0])
        
        # Pickup urgency signals (3 values): [health_urgency, machinegun_urgency, ammo_urgency]
        urgency = compute_pickup_urgency()
        state.extend([
            urgency['health'],
            urgency['machinegun'],
            urgency['ammo']
        ])
        
        # Zombie counts (2 values): [in_bounds_count_normalized, total_count_normalized]
        in_bounds_count = len(in_bounds_zombies)
        total_count = len(game.zombies)
        state.extend([
            min(in_bounds_count / 15.0, 1.0),
            min(total_count / 20.0, 1.0)
        ])
        
        # Verify state dimension
        assert len(state) == 129, f"State dimension mismatch: expected 129, got {len(state)}"
        
        return np.array(state, dtype=np.float32)
    
    def get_nearest_pickup(pickup_type):
        """Get nearest pickup of given type."""
        pickups_of_type = [p for p in game.pickups if p.pickup_type == pickup_type]
        if not pickups_of_type:
            return None
        return min(pickups_of_type, 
                   key=lambda p: math.hypot(game.player.x - p.x, game.player.y - p.y))
    
    state = get_agent_state()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if game.game_over and event.key == K_r:
                    # FIXED: Use resetGame() for complete reset
                    game.resetGame()
                    episode_reward = 0
                    episode_steps = 0
                    state = get_agent_state()
                    print(f"\nEpisode {episodes} - Restarting...")
        
        if not game.game_over:
            # Get action from agent (multi-discrete: combined action)
            combined_action, _, _ = agent.select_action(state, deterministic=True)
            
            # Decode multi-discrete action
            movement_action, shoot_action = agent.decode_action(combined_action)
            
            # Apply movement (5 actions: 0-3=Move, 4=Idle)
            if movement_action == 0:  # Move Up
                game.player.y -= PLAYER_SPEED
            elif movement_action == 1:  # Move Down
                game.player.y += PLAYER_SPEED
            elif movement_action == 2:  # Move Left
                game.player.x -= PLAYER_SPEED
            elif movement_action == 3:  # Move Right
                game.player.x += PLAYER_SPEED
            # movement_action == 4 is Idle, do nothing
            
            # Apply shooting (independent of movement)
            if shoot_action == 1:  # Shoot
                # Auto-select best weapon: prefer machine gun if available and has ammo
                if (game.player.has_machinegun and 
                    game.player.machinegun_ammo > 0 and 
                    game.player.weapon != "machinegun"):
                    game.player.weapon = "machinegun"
                elif (not game.player.has_machinegun or 
                      game.player.machinegun_ammo <= 0):
                    game.player.weapon = "pistol"
                
                if game.player.shoot_cooldown == 0:
                    if game.player.weapon == "machinegun" and game.player.has_machinegun and game.player.machinegun_ammo > 0:
                        game.bullets.append(Bullet(game.player.x, game.player.y, game.player.angle, "machinegun"))
                        game.player.shoot_cooldown = MACHINEGUN_COOLDOWN
                        game.player.machinegun_ammo -= 1
                        if game.player.machinegun_ammo <= 0:
                            game.player.has_machinegun = False
                            game.player.weapon = "pistol"
                    else:
                        game.bullets.append(Bullet(game.player.x, game.player.y, game.player.angle, "pistol"))
                        game.player.shoot_cooldown = PISTOL_COOLDOWN
            
            # Keep player within arena bounds (glass borders prevent movement outside)
            game.player.x = max(ARENA_X_OFFSET, min(ARENA_X_OFFSET + ARENA_WIDTH, game.player.x))
            game.player.y = max(ARENA_Y_OFFSET, min(ARENA_Y_OFFSET + ARENA_HEIGHT, game.player.y))
            
            # Update cooldown
            if game.player.shoot_cooldown > 0:
                game.player.shoot_cooldown -= 1
            
            # Update player angle to nearest IN-BOUNDS zombie only
            in_bounds_zombies = get_in_bounds_zombies()
            if len(in_bounds_zombies) > 0:
                nearest_zombie = min(in_bounds_zombies, 
                                   key=lambda z: math.hypot(game.player.x - z.x, game.player.y - z.y))
                dx = nearest_zombie.x - game.player.x
                dy = nearest_zombie.y - game.player.y
                game.player.angle = math.degrees(math.atan2(-dy, dx)) % 360
            # If no in-bounds zombies, keep previous angle
            
            # Update pickups
            pickups_to_remove = []
            for pickup in game.pickups:
                if pickup.update():
                    pickups_to_remove.append(pickup)
                elif pickup.check_collision(game.player.x, game.player.y):
                    if pickup.pickup_type == "health":
                        game.player.add_health(HEALTH_PICKUP_AMOUNT)
                    elif pickup.pickup_type == "machinegun":
                        game.player.pickup_machinegun(MACHINEGUN_PICKUP_AMMO)
                    elif pickup.pickup_type == "ammo":
                        # Ammo pickup for machine gun
                        if game.player.has_machinegun:
                            game.player.machinegun_ammo += 30  # AMMO_PICKUP_AMOUNT
                    pickups_to_remove.append(pickup)
            for pickup in pickups_to_remove:
                if pickup in game.pickups:
                    game.pickups.remove(pickup)
            
            # Update bullets
            bullets_to_remove = []
            for bullet in game.bullets:
                if bullet.update():
                    bullets_to_remove.append(bullet)
                else:
                    zombies_to_remove = []
                    for zombie in game.zombies:
                        if bullet.check_collision(zombie):
                            if zombie.take_damage(bullet.damage):
                                zombies_to_remove.append(zombie)
                                game.player.score += 10 if zombie.zombie_type == "normal" else 20
                                game.zombies_killed += 1
                                episode_reward += 10
                                drop_chance = random.random()
                                if drop_chance < HEALTH_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "health"))
                                elif drop_chance < HEALTH_PICKUP_DROP_PROBABILITY + MACHINEGUN_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "machinegun"))
                                elif drop_chance < (HEALTH_PICKUP_DROP_PROBABILITY + 
                                                   MACHINEGUN_PICKUP_DROP_PROBABILITY + 0.20):
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "ammo"))
                            bullets_to_remove.append(bullet)
                            break
                    for zombie in zombies_to_remove:
                        game.zombies.remove(zombie)
            for bullet in bullets_to_remove:
                if bullet in game.bullets:
                    game.bullets.remove(bullet)
            
            # Spawn zombies with wave progression
            game.spawn_timer += 1
            if game.spawn_timer >= game.spawn_rate:
                game.zombies.append(spawn_zombie())
                game.spawn_timer = 0
                if game.zombies_killed > 0 and game.zombies_killed % 10 == 0:
                    expected_wave = (game.zombies_killed // 10) + 1
                    if expected_wave > game.wave:
                        game.wave = expected_wave
                        game.spawn_rate = max(10, game.spawn_rate - 5)
            
            # Update zombies
            for zombie in game.zombies:
                zombie.update(game.player.x, game.player.y)
                if zombie.check_collision_with_player(game.player.x, game.player.y):
                    if zombie.can_attack():
                        damage = zombie.attack()
                        if game.player.take_damage(damage):
                            game.game_over = True
                            episodes += 1
                            print(f"\n{'='*50}")
                            print(f"Episode {episodes} finished!")
                            print(f"Total Reward: {episode_reward:.2f}")
                            print(f"Steps: {episode_steps}")
                            print(f"Zombies Killed: {game.zombies_killed}")
                            print(f"Wave Reached: {game.wave}")
                            print(f"Weapon: {game.player.weapon}")
                            print("Result: DIED")
                            print(f"{'='*50}\n")
            
            # Update state
            state = get_agent_state()
            episode_steps += 1
            episode_reward += 0.1
            
            if episode_steps % 100 == 0:
                print(f"Steps: {episode_steps}, Reward: {episode_reward:.2f}, " +
                      f"Kills: {game.zombies_killed}, Health: {game.player.health}, " +
                      f"Wave: {game.wave}, Weapon: {game.player.weapon}")
        
        # Draw everything with FULL ASSETS
        # Fill expanded screen background
        screen.fill(BG_COLOR)
        
        # Draw arena background (playable area)
        arena_rect = pygame.Rect(
            ARENA_X_OFFSET,
            ARENA_Y_OFFSET,
            ARENA_WIDTH,
            ARENA_HEIGHT
        )
        if background_img:
            screen.blit(background_img, (ARENA_X_OFFSET, ARENA_Y_OFFSET))
        else:
            pygame.draw.rect(screen, (80, 80, 80), arena_rect)
        
        # Draw glass borders around the arena (only on left and right sides as requested)
        glass_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Left glass border (full height of arena)
        left_border = pygame.Rect(
            ARENA_X_OFFSET - GLASS_BORDER_THICKNESS,
            ARENA_Y_OFFSET,
            GLASS_BORDER_THICKNESS,
            ARENA_HEIGHT
        )
        pygame.draw.rect(glass_surface, GLASS_COLOR, left_border)
        
        # Right glass border (full height of arena)
        right_border = pygame.Rect(
            ARENA_X_OFFSET + ARENA_WIDTH,
            ARENA_Y_OFFSET,
            GLASS_BORDER_THICKNESS,
            ARENA_HEIGHT
        )
        pygame.draw.rect(glass_surface, GLASS_COLOR, right_border)
        
        # Blit the glass surface onto the screen
        screen.blit(glass_surface, (0, 0))
        
        game.player.draw(screen)
        for zombie in game.zombies:
            zombie.draw(screen)
        for bullet in game.bullets:
            bullet.draw(screen)
        for knife in game.knife_attacks:
            knife.draw(screen)
        for pickup in game.pickups:
            pickup.draw(screen)
        
        draw_ui(screen, game.player, game.wave, game.zombies_killed)
        
        if game.game_over:
            draw_game_over(screen, game.player.score, game.zombies_killed, game.wave)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


def main():
    """Main entry point with mode selection."""
    parser = argparse.ArgumentParser(description="Top-Down Zombie Shooter")
    parser.add_argument('--mode', type=str, default='manual', choices=['manual', 'agent'],
                       help='Game mode: manual (keyboard/mouse) or agent (PPO plays)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model (for agent mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        print("Starting in MANUAL mode (player control)")
        main_manual_mode()
    elif args.mode == 'agent':
        print("Starting in AGENT mode (PPO control)")
        main_agent_mode(args.model)


if __name__ == "__main__":
    main()
