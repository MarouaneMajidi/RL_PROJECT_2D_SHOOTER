import os
import pygame
import sys
import math
import random
from pygame.locals import *

# Asset constants
ASSET_DIR = os.path.join(os.path.dirname(__file__), "topdown")

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
PLAYER_SPEED = 5
ZOMBIE_SPEED = 2
BULLET_SPEED = 10
SPAWN_RATE = 60  # Frames between zombie spawns

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
background_img = pygame.transform.scale(background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

class Player:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.angle = 0
        self.weapon = "pistol"  # pistol, machinegun, knife
        self.health = 100
        self.score = 0
        self.shoot_cooldown = 0
        self.knife_cooldown = 0
        self.speed = PLAYER_SPEED
        
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
            
        # Keep player on screen
        self.x = max(0, min(SCREEN_WIDTH, self.x))
        self.y = max(0, min(SCREEN_HEIGHT, self.y))
        
        # Weapon switching
        if keys[K_1]:
            self.weapon = "pistol"
        if keys[K_2]:
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
                self.shoot_cooldown = 20  # Cooldown in frames
                return Bullet(self.x, self.y, self.angle, "pistol")
            elif self.weapon == "machinegun":
                self.shoot_cooldown = 5
                return Bullet(self.x, self.y, self.angle, "machinegun")
                
        # Knife attack
        if mouse_buttons[0] and self.knife_cooldown == 0 and self.weapon == "knife":
            self.knife_cooldown = 30
            return KnifeAttack(self.x, self.y, self.angle)
            
        return None
        
    def draw(self, screen):
        # Draw the appropriate player image based on weapon
        if self.weapon == "pistol":
            img = player_pistol_img
        elif self.weapon == "machinegun":
            img = player_machinegun_img
        else:  # knife
            img = player_knife_img
            
        # Rotate image to face mouse
        rotated_img = pygame.transform.rotate(img, self.angle)
        rect = rotated_img.get_rect(center=(self.x, self.y))
        screen.blit(rotated_img, rect)
        
        # Draw health bar
        pygame.draw.rect(screen, RED, (self.x - 25, self.y - 40, 50, 5))
        pygame.draw.rect(screen, GREEN, (self.x - 25, self.y - 40, 50 * (self.health / 100), 5))
        
    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

class Zombie:
    def __init__(self, x, y, zombie_type="normal"):
        self.x = x
        self.y = y
        self.zombie_type = zombie_type
        self.speed = ZOMBIE_SPEED
        self.health = 30 if zombie_type == "normal" else 50
        self.damage = 10 if zombie_type == "normal" else 20
        self.base_image = zombie_img if zombie_type == "normal" else zombie2_img
        self.angle = 0
        
    def update(self, player_x, player_y):
        # Move towards player
        dx = player_x - self.x
        dy = player_y - self.y
        dist = max(0.1, math.hypot(dx, dy))  # Avoid division by zero
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        self.angle = math.degrees(math.atan2(-dy, dx)) % 360
        
    def draw(self, screen):
        rotated = pygame.transform.rotate(self.base_image, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)
        
        # Draw health bar
        health_width = 30
        pygame.draw.rect(screen, RED, (self.x - health_width//2, self.y - 30, health_width, 5))
        max_health = 30 if self.zombie_type == "normal" else 50
        pygame.draw.rect(screen, GREEN, (self.x - health_width//2, self.y - 30, health_width * (self.health / max_health), 5))
        
    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0
        
    def check_collision_with_player(self, player_x, player_y):
        # Simple circle collision
        distance = math.hypot(player_x - self.x, player_y - self.y)
        return distance < 30

class Bullet:
    def __init__(self, x, y, angle, weapon_type):
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.weapon_type = weapon_type
        self.speed = BULLET_SPEED
        self.damage = 20 if weapon_type == "pistol" else 10
        
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y -= math.sin(self.angle) * self.speed
        
        # Check if bullet is off screen
        return (self.x < 0 or self.x > SCREEN_WIDTH or 
                self.y < 0 or self.y > SCREEN_HEIGHT)
        
    def draw(self, screen):
        color = YELLOW if self.weapon_type == "pistol" else (255, 165, 0)  # Orange for machinegun
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 4)
        
    def check_collision(self, zombie):
        distance = math.hypot(self.x - zombie.x, self.y - zombie.y)
        return distance < 25

class KnifeAttack:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.range = 50
        self.damage = 40
        self.lifetime = 5  # Frames
        
    def update(self):
        self.lifetime -= 1
        return self.lifetime <= 0
        
    def draw(self, screen):
        # Draw knife attack arc
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
            
        # Check if zombie is within the arc
        dx = zombie.x - self.x
        dy = zombie.y - self.y
        zombie_angle = math.atan2(-dy, dx) % (2 * math.pi)
        attack_angle = self.angle % (2 * math.pi)
        
        angle_diff = abs(zombie_angle - attack_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        
        return angle_diff <= math.radians(45)

def spawn_zombie():
    # Spawn zombies from outside the screen
    side = random.randint(0, 3)
    if side == 0:  # Top
        x = random.randint(0, SCREEN_WIDTH)
        y = -50
    elif side == 1:  # Right
        x = SCREEN_WIDTH + 50
        y = random.randint(0, SCREEN_HEIGHT)
    elif side == 2:  # Bottom
        x = random.randint(0, SCREEN_WIDTH)
        y = SCREEN_HEIGHT + 50
    else:  # Left
        x = -50
        y = random.randint(0, SCREEN_HEIGHT)
        
    # 20% chance for a stronger zombie
    zombie_type = "strong" if random.random() < 0.2 else "normal"
    return Zombie(x, y, zombie_type)

def draw_ui(screen, player, wave, zombies_killed):
    # Draw score and weapon info
    font = pygame.font.SysFont(None, 36)
    
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    
    wave_text = font.render(f"Wave: {wave}", True, WHITE)
    screen.blit(wave_text, (10, 50))
    
    kills_text = font.render(f"Zombies Killed: {zombies_killed}", True, WHITE)
    screen.blit(kills_text, (10, 90))
    
    # Draw weapon info
    weapon_text = font.render(f"Weapon: {player.weapon.capitalize()}", True, WHITE)
    screen.blit(weapon_text, (SCREEN_WIDTH - 200, 10))
    
    # Draw weapon keys
    keys_text = font.render("1:Pistol  2:Machine Gun  3:Knife", True, WHITE)
    screen.blit(keys_text, (SCREEN_WIDTH - 300, 50))
    
    # Draw health text
    health_text = font.render(f"Health: {player.health}", True, WHITE)
    screen.blit(health_text, (SCREEN_WIDTH - 150, 90))

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

def main():
    # Use a local variable for spawn rate instead of global
    spawn_rate = SPAWN_RATE
    
    player = Player()
    zombies = []
    bullets = []
    knife_attacks = []
    
    zombies_killed = 0
    wave = 1
    spawn_timer = 0
    game_over = False
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if game_over and event.key == K_r:
                    # Restart game
                    return main()
                if game_over and event.key == K_q:
                    pygame.quit()
                    sys.exit()
        
        if not game_over:
            # Get input
            keys = pygame.key.get_pressed()
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            # Update player
            new_bullet = player.update(keys, mouse_pos, mouse_buttons)
            if new_bullet:
                if isinstance(new_bullet, Bullet):
                    bullets.append(new_bullet)
                else:  # KnifeAttack
                    knife_attacks.append(new_bullet)
            
            # Update bullets
            bullets_to_remove = []
            for bullet in bullets:
                if bullet.update():
                    bullets_to_remove.append(bullet)
                else:
                    # Check bullet collisions with zombies
                    zombies_to_remove = []
                    for zombie in zombies:
                        if bullet.check_collision(zombie):
                            if zombie.take_damage(bullet.damage):
                                zombies_to_remove.append(zombie)
                                player.score += 10 if zombie.zombie_type == "normal" else 20
                                zombies_killed += 1
                            bullets_to_remove.append(bullet)
                            break
                    
                    # Remove dead zombies
                    for zombie in zombies_to_remove:
                        zombies.remove(zombie)
            
            # Remove bullets that are off screen or hit a zombie
            for bullet in bullets_to_remove:
                if bullet in bullets:
                    bullets.remove(bullet)
            
            # Update knife attacks
            knife_attacks_to_remove = []
            for knife in knife_attacks:
                if knife.update():
                    knife_attacks_to_remove.append(knife)
                else:
                    # Check knife collisions with zombies
                    zombies_to_remove = []
                    for zombie in zombies:
                        if knife.check_collision(zombie):
                            if zombie.take_damage(knife.damage):
                                zombies_to_remove.append(zombie)
                                player.score += 10 if zombie.zombie_type == "normal" else 20
                                zombies_killed += 1
                    
                    # Remove dead zombies
                    for zombie in zombies_to_remove:
                        zombies.remove(zombie)
            
            # Remove expired knife attacks
            for knife in knife_attacks_to_remove:
                if knife in knife_attacks:
                    knife_attacks.remove(knife)
            
            # Spawn zombies
            spawn_timer += 1
            if spawn_timer >= spawn_rate:
                zombies.append(spawn_zombie())
                spawn_timer = 0
                
                # Increase spawn rate every 10 zombies killed
                if zombies_killed > 0 and zombies_killed % 10 == 0:
                    spawn_rate = max(10, spawn_rate - 5)
                    wave += 1
            
            # Update zombies
            for zombie in zombies:
                zombie.update(player.x, player.y)
                
                # Check for zombie-player collision
                if zombie.check_collision_with_player(player.x, player.y):
                    if player.take_damage(zombie.damage):
                        game_over = True
        
        # Draw everything
        # Draw background image (falls back to solid fill if needed)
        if background_img:
            screen.blit(background_img, (0, 0))
        else:
            screen.fill(BG_COLOR)
        
        # Draw player
        player.draw(screen)
        
        # Draw zombies
        for zombie in zombies:
            zombie.draw(screen)
        
        # Draw bullets
        for bullet in bullets:
            bullet.draw(screen)
            
        # Draw knife attacks
        for knife in knife_attacks:
            knife.draw(screen)
        
        # Draw UI
        draw_ui(screen, player, wave, zombies_killed)
        
        # Draw game over screen if needed
        if game_over:
            draw_game_over(screen, player.score, zombies_killed, wave)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()