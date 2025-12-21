"""
Play the game with DDA agent adjusting difficulty.

You control the player manually while the DDA agent dynamically adjusts
game difficulty based on your performance.
"""

import os
import sys
import pygame
import math
import random
from pygame.locals import *

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from dda_agent import DDAAgent, DDAConfig, DDAStateExtractor, DifficultyManager

# Asset constants
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Initialize pygame
pygame.init()

# Game constants
ARENA_WIDTH = 800
ARENA_HEIGHT = 600
SCREEN_WIDTH = 850
SCREEN_HEIGHT = 650
ARENA_X_OFFSET = (SCREEN_WIDTH - ARENA_WIDTH) // 2
ARENA_Y_OFFSET = (SCREEN_HEIGHT - ARENA_HEIGHT) // 2

FPS = 60

# Base difficulty parameters (will be modified by DDA)
BASE_PARAMS = {
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

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
BG_COLOR = (96, 96, 96)
GLASS_COLOR = (200, 200, 255, 180)

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Zombie Shooter - DDA Mode (You Play!)")
clock = pygame.time.Clock()

# Load sprites
def load_sprite(filename, scale=1.0):
    path = os.path.join(ASSET_DIR, filename)
    image = pygame.image.load(path).convert_alpha()
    if scale != 1.0:
        new_size = (int(image.get_width() * scale), int(image.get_height() * scale))
        image = pygame.transform.scale(image, new_size)
    return image

player_pistol_img = load_sprite("player pistol.gif", 1.2)
player_machinegun_img = load_sprite("player machinegun.gif", 1.2)
player_knife_img = load_sprite("player knife.gif", 1.2)
zombie_img = load_sprite("zombie.gif", 1.2)
zombie2_img = load_sprite("zombie 2.gif", 1.2)
background_img = pygame.image.load(os.path.join(ASSET_DIR, "background.png")).convert()
background_img = pygame.transform.scale(background_img, (ARENA_WIDTH, ARENA_HEIGHT))


class Player:
    def __init__(self, params):
        self.params = params
        self.reset()
        
    def reset(self):
        self.x = ARENA_X_OFFSET + ARENA_WIDTH // 2
        self.y = ARENA_Y_OFFSET + ARENA_HEIGHT // 2
        self.angle = 0
        self.weapon = "pistol"
        self.health = 100
        self.score = 0
        self.shoot_cooldown = 0
        self.has_machinegun = False
        self.machinegun_ammo = 0
        
    def update(self, keys, mouse_pos, mouse_buttons):
        # Angle to mouse
        dx = mouse_pos[0] - self.x
        dy = mouse_pos[1] - self.y
        self.angle = math.degrees(math.atan2(-dy, dx)) % 360
        
        # Movement
        speed = self.params['PLAYER_SPEED']
        if keys[K_w] or keys[K_UP]:
            self.y -= speed
        if keys[K_s] or keys[K_DOWN]:
            self.y += speed
        if keys[K_a] or keys[K_LEFT]:
            self.x -= speed
        if keys[K_d] or keys[K_RIGHT]:
            self.x += speed
            
        # Bounds
        self.x = max(ARENA_X_OFFSET, min(ARENA_X_OFFSET + ARENA_WIDTH, self.x))
        self.y = max(ARENA_Y_OFFSET, min(ARENA_Y_OFFSET + ARENA_HEIGHT, self.y))
        
        # Weapon switching
        if keys[K_1]:
            self.weapon = "pistol"
        if keys[K_2] and self.has_machinegun:
            self.weapon = "machinegun"
        if keys[K_3]:
            self.weapon = "knife"
            
        # Cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            
        # Shooting
        if mouse_buttons[0] and self.shoot_cooldown == 0 and self.weapon != "knife":
            if self.weapon == "pistol":
                self.shoot_cooldown = int(self.params['PISTOL_COOLDOWN'])
                return Bullet(self.x, self.y, self.angle, "pistol", self.params)
            elif self.weapon == "machinegun" and self.has_machinegun and self.machinegun_ammo > 0:
                self.shoot_cooldown = int(self.params['MACHINEGUN_COOLDOWN'])
                self.machinegun_ammo -= 1
                if self.machinegun_ammo <= 0:
                    self.has_machinegun = False
                    self.weapon = "pistol"
                return Bullet(self.x, self.y, self.angle, "machinegun", self.params)
        return None
        
    def draw(self, screen):
        if self.weapon == "pistol":
            img = player_pistol_img
        elif self.weapon == "machinegun" and self.has_machinegun:
            img = player_machinegun_img
        else:
            img = player_knife_img if self.weapon == "knife" else player_pistol_img
            
        rotated_img = pygame.transform.rotate(img, self.angle)
        rect = rotated_img.get_rect(center=(self.x, self.y))
        screen.blit(rotated_img, rect)
        
        # Health bar
        pygame.draw.rect(screen, RED, (self.x - 25, self.y - 40, 50, 5))
        pygame.draw.rect(screen, GREEN, (self.x - 25, self.y - 40, 50 * (self.health / 100), 5))


class Zombie:
    def __init__(self, x, y, zombie_type, params):
        self.x = x
        self.y = y
        self.zombie_type = zombie_type
        self.params = params
        self.speed = params['ZOMBIE_SPEED']
        self.health = params['ZOMBIE_NORMAL_HEALTH'] if zombie_type == "normal" else params['ZOMBIE_STRONG_HEALTH']
        self.max_health = self.health
        self.damage = 10 if zombie_type == "normal" else 20
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
        
    def draw(self, screen):
        rotated = pygame.transform.rotate(self.base_image, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)
        
        # Health bar
        pygame.draw.rect(screen, RED, (self.x - 15, self.y - 30, 30, 5))
        pygame.draw.rect(screen, GREEN, (self.x - 15, self.y - 30, 30 * (self.health / self.max_health), 5))


class Bullet:
    def __init__(self, x, y, angle, weapon_type, params):
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.weapon_type = weapon_type
        self.speed = 10
        self.damage = params['PISTOL_DAMAGE'] if weapon_type == "pistol" else params['MACHINEGUN_DAMAGE']
        
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y -= math.sin(self.angle) * self.speed
        return (self.x < 0 or self.x > SCREEN_WIDTH or 
                self.y < 0 or self.y > SCREEN_HEIGHT)
        
    def draw(self, screen):
        color = YELLOW if self.weapon_type == "pistol" else (255, 165, 0)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 4)


class Pickup:
    def __init__(self, x, y, pickup_type):
        self.x = x
        self.y = y
        self.pickup_type = pickup_type
        self.lifetime = 600
        
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
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), 18, 2)


def spawn_zombie(params):
    side = random.randint(0, 3)
    if side == 0:
        x = random.uniform(ARENA_X_OFFSET, ARENA_X_OFFSET + ARENA_WIDTH)
        y = ARENA_Y_OFFSET
    elif side == 1:
        x = ARENA_X_OFFSET + ARENA_WIDTH
        y = random.uniform(ARENA_Y_OFFSET, ARENA_Y_OFFSET + ARENA_HEIGHT)
    elif side == 2:
        x = random.uniform(ARENA_X_OFFSET, ARENA_X_OFFSET + ARENA_WIDTH)
        y = ARENA_Y_OFFSET + ARENA_HEIGHT
    else:
        x = ARENA_X_OFFSET
        y = random.uniform(ARENA_Y_OFFSET, ARENA_Y_OFFSET + ARENA_HEIGHT)
    zombie_type = "strong" if random.random() < 0.2 else "normal"
    return Zombie(x, y, zombie_type, params)


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


def main(dda_model_path: str = "checkpoints/dda/best_dda_model.pth"):
    """Run game with DDA."""
    
    # Load DDA agent
    print("Loading DDA agent...")
    dda_config = DDAConfig()
    dda_agent = DDAAgent(dda_config)
    
    try:
        dda_agent.load(dda_model_path)
        print("DDA agent loaded!")
        use_dda = True
    except FileNotFoundError:
        print(f"Warning: DDA model not found at {dda_model_path}")
        print("Running without DDA (base difficulty)")
        use_dda = False
    
    # Initialize DDA components
    state_extractor = DDAStateExtractor(window_size=600, fps=60)
    difficulty_manager = DifficultyManager(BASE_PARAMS)
    
    # Game state
    params = difficulty_manager.get_current_params()
    player = Player(params)
    zombies = []
    bullets = []
    pickups = []
    
    spawn_timer = 0
    zombies_killed = 0
    dda_action_timer = 0
    dda_action_interval = 300  # 5 seconds
    last_dda_action = "Initializing..."  # Will be updated when DDA acts
    previous_dda_action = "None"  # Previous decision
    
    # Metrics tracking
    last_health = 100
    
    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_r:
                    # Reset
                    player = Player(params)
                    zombies = []
                    bullets = []
                    pickups = []
                    spawn_timer = 0
                    zombies_killed = 0
                    state_extractor.reset()
                    difficulty_manager.reset()
                    params = difficulty_manager.get_current_params()
                    last_health = 100
        
        if player.health > 0:
            # Input
            keys = pygame.key.get_pressed()
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            # Update player
            new_bullet = player.update(keys, mouse_pos, mouse_buttons)
            if new_bullet:
                bullets.append(new_bullet)
            
            # Track metrics
            damage_taken = max(0, last_health - player.health)
            last_health = player.health
            
            state_extractor.update({
                'player_health': player.health,
                'damage_taken': damage_taken,
                'damage_dealt': 0,
                'zombie_killed': False,
                'shot_fired': new_bullet is not None,
                'shot_hit': False,
                'pickup_collected': False,
            })
            
            # DDA action
            dda_action_timer += 1
            if use_dda and dda_action_timer >= dda_action_interval:
                dda_action_timer = 0
                
                # Extract state
                dda_state = state_extractor.extract_state({
                    'player_health': player.health,
                    'zombies_alive': len(zombies),
                    'max_zombies': 20,
                    'machinegun_ammo': player.machinegun_ammo,
                    'max_ammo': 100,
                })
                
                # Get DDA action
                action, _, _ = dda_agent.select_action(dda_state, deterministic=True)
                
                # Apply action
                difficulty_manager.apply_action(action)
                params = difficulty_manager.get_current_params()
                player.params = params
                
                # Update zombie speeds
                for z in zombies:
                    z.speed = params['ZOMBIE_SPEED']
                
                action_names = ["Do Nothing", "Slightly Easier", "Much Easier", "Slightly Harder", "Much Harder"]
                # Update previous action before setting new one
                if last_dda_action != "Initializing...":
                    previous_dda_action = last_dda_action
                last_dda_action = action_names[action]
                print(f"DDA Action: {last_dda_action} | Zombie Speed: {params['ZOMBIE_SPEED']:.2f} | Spawn Rate: {params['SPAWN_RATE']:.0f}")
            
            # Update pickups
            pickups_to_remove = []
            for pickup in pickups:
                if pickup.update():
                    pickups_to_remove.append(pickup)
                elif math.hypot(player.x - pickup.x, player.y - pickup.y) < 25:
                    if pickup.pickup_type == "health":
                        player.health = min(100, player.health + 25)
                    elif pickup.pickup_type == "machinegun":
                        player.has_machinegun = True
                        player.machinegun_ammo += 50
                    pickups_to_remove.append(pickup)
            for p in pickups_to_remove:
                if p in pickups:
                    pickups.remove(p)
            
            # Update bullets
            bullets_to_remove = []
            for bullet in bullets:
                if bullet.update():
                    bullets_to_remove.append(bullet)
                else:
                    for zombie in zombies[:]:
                        if math.hypot(bullet.x - zombie.x, bullet.y - zombie.y) < 25:
                            zombie.health -= bullet.damage
                            if zombie.health <= 0:
                                zombies.remove(zombie)
                                zombies_killed += 1
                                player.score += 10
                                # Drop pickup
                                if random.random() < params['HEALTH_PICKUP_DROP_PROBABILITY']:
                                    pickups.append(Pickup(zombie.x, zombie.y, "health"))
                                elif random.random() < params['MACHINEGUN_PICKUP_DROP_PROBABILITY']:
                                    pickups.append(Pickup(zombie.x, zombie.y, "machinegun"))
                            bullets_to_remove.append(bullet)
                            break
            for b in bullets_to_remove:
                if b in bullets:
                    bullets.remove(b)
            
            # Spawn zombies
            spawn_timer += 1
            if spawn_timer >= params['SPAWN_RATE']:
                zombies.append(spawn_zombie(params))
                spawn_timer = 0
            
            # Update zombies
            for zombie in zombies:
                zombie.update(player.x, player.y)
                if math.hypot(zombie.x - player.x, zombie.y - player.y) < 30:
                    if zombie.attack_cooldown == 0:
                        player.health -= zombie.damage
                        zombie.attack_cooldown = int(params['ZOMBIE_ATTACK_COOLDOWN'])
        
        # Draw
        screen.fill(BG_COLOR)
        screen.blit(background_img, (ARENA_X_OFFSET, ARENA_Y_OFFSET))
        
        player.draw(screen)
        for zombie in zombies:
            zombie.draw(screen)
        for bullet in bullets:
            bullet.draw(screen)
        for pickup in pickups:
            pickup.draw(screen)
        
        # UI - Left side
        font = pygame.font.SysFont(None, 36)
        font_small = pygame.font.SysFont(None, 24)
        screen.blit(font.render(f"Health: {player.health}", True, WHITE), (10, 10))
        screen.blit(font.render(f"Score: {player.score}", True, WHITE), (10, 50))
        screen.blit(font.render(f"Kills: {zombies_killed}", True, WHITE), (10, 90))
        screen.blit(font.render(f"Zombies: {len(zombies)}", True, WHITE), (10, 130))
        
        # UI - Right side: DDA Actions (fixed, always visible, updates when DDA acts)
        if use_dda:
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
        
        if player.health <= 0:
            # Game over
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            
            font_large = pygame.font.SysFont(None, 72)
            screen.blit(font_large.render("GAME OVER", True, RED), (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 50))
            screen.blit(font.render("Press R to Restart", True, WHITE), (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play game with DDA (Manual Mode)")
    parser.add_argument('--model', type=str, default='checkpoints/dda/best_dda_model.pth',
                       help='Path to DDA model')
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("ZOMBIE SHOOTER - DDA MODE (YOU PLAY)")
    print("="*50)
    print("Controls:")
    print("  WASD/Arrows - Move")
    print("  Mouse - Aim")
    print("  Left Click - Shoot")
    print("  1/2/3 - Switch weapons")
    print("  R - Restart")
    print("  ESC - Quit")
    print("\nThe DDA agent adjusts difficulty every 5 seconds!")
    print("Watch the top-right corner for DDA actions.")
    print("="*50 + "\n")
    
    main(args.model)

