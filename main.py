import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1280, 720
FPS = 60

# Colors
SAND = (194, 178, 128)
DARK_BLUE = (30, 60, 90)
PINK = (255, 180, 180)
RED = (220, 50, 50)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Game settings
PLAYER_SPEED = 5
PLAYER_MAX_HEALTH = 100
BULLET_SPEED = 10
ENEMY_SPEED = 2
SPAWN_RATE = 60  # frames between spawns

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((40, 40), pygame.SRCALPHA)
        # Draw player (blue circle with pink claws)
        pygame.draw.circle(self.image, DARK_BLUE, (20, 20), 15)
        pygame.draw.circle(self.image, PINK, (8, 12), 6)
        pygame.draw.circle(self.image, PINK, (32, 12), 6)
        self.rect = self.image.get_rect(center=(x, y))
        self.health = PLAYER_MAX_HEALTH
        self.shoot_cooldown = 0
        
    def update(self, keys):
        # Movement
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.rect.y -= PLAYER_SPEED
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.rect.y += PLAYER_SPEED
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.rect.x -= PLAYER_SPEED
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.rect.x += PLAYER_SPEED
            
        # Keep player on screen
        self.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))
        
        # Update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def shoot(self, target_pos):
        if self.shoot_cooldown == 0:
            self.shoot_cooldown = 15
            return Bullet(self.rect.centerx, self.rect.centery, target_pos)
        return None

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, target_pos):
        super().__init__()
        self.image = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(self.image, DARK_BLUE, (4, 4), 4)
        self.rect = self.image.get_rect(center=(x, y))
        
        # Calculate direction
        dx = target_pos[0] - x
        dy = target_pos[1] - y
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            self.vel_x = (dx / dist) * BULLET_SPEED
            self.vel_y = (dy / dist) * BULLET_SPEED
        else:
            self.vel_x = 0
            self.vel_y = 0
    
    def update(self):
        self.rect.x += self.vel_x
        self.rect.y += self.vel_y
        
        # Remove if off screen
        if not pygame.Rect(0, 0, WIDTH, HEIGHT).colliderect(self.rect):
            self.kill()

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((35, 35), pygame.SRCALPHA)
        # Draw enemy (blue body with pink claws)
        pygame.draw.circle(self.image, DARK_BLUE, (17, 20), 12)
        # Claws
        pygame.draw.circle(self.image, PINK, (5, 8), 5)
        pygame.draw.circle(self.image, PINK, (29, 8), 5)
        pygame.draw.circle(self.image, PINK, (5, 25), 5)
        pygame.draw.circle(self.image, PINK, (29, 25), 5)
        
        self.rect = self.image.get_rect(center=(x, y))
        self.health = 2
        
    def update(self, player_pos):
        # Move towards player
        dx = player_pos[0] - self.rect.centerx
        dy = player_pos[1] - self.rect.centery
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            self.rect.x += (dx / dist) * ENEMY_SPEED
            self.rect.y += (dy / dist) * ENEMY_SPEED

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Survival Shooter")
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.spawn_timer = 0
        self.spawn_rate = SPAWN_RATE
        
        # Sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        
        # Create player
        self.player = Player(WIDTH // 2, HEIGHT // 2)
        self.all_sprites.add(self.player)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def spawn_enemy(self):
        # Spawn from edges
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            x, y = random.randint(0, WIDTH), -20
        elif side == 'bottom':
            x, y = random.randint(0, WIDTH), HEIGHT + 20
        elif side == 'left':
            x, y = -20, random.randint(0, HEIGHT)
        else:
            x, y = WIDTH + 20, random.randint(0, HEIGHT)
            
        enemy = Enemy(x, y)
        self.enemies.add(enemy)
        self.all_sprites.add(enemy)
    
    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()
            
        pygame.quit()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    bullet = self.player.shoot(event.pos)
                    if bullet:
                        self.bullets.add(bullet)
                        self.all_sprites.add(bullet)
    
    def update(self):
        # Get keys
        keys = pygame.key.get_pressed()
        
        # Update player
        self.player.update(keys)
        
        # Update bullets
        self.bullets.update()
        
        # Update enemies
        self.enemies.update(self.player.rect.center)
        
        # Spawn enemies
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_enemy()
            self.spawn_timer = 0
            # Increase difficulty
            if self.score > 50:
                self.spawn_rate = max(20, 60 - self.score // 10)
        
        # Check bullet-enemy collisions
        for bullet in self.bullets:
            hit_enemies = pygame.sprite.spritecollide(bullet, self.enemies, False)
            if hit_enemies:
                bullet.kill()
                for enemy in hit_enemies:
                    enemy.health -= 1
                    if enemy.health <= 0:
                        enemy.kill()
                        self.score += 10
        
        # Check player-enemy collisions
        hit_enemies = pygame.sprite.spritecollide(self.player, self.enemies, True)
        if hit_enemies:
            self.player.health -= len(hit_enemies) * 10
            if self.player.health <= 0:
                self.game_over()
    
    def draw(self):
        # Background
        self.screen.fill(SAND)
        
        # Draw all sprites
        self.all_sprites.draw(self.screen)
        
        # Draw UI
        # Health bar
        health_width = int((self.player.health / PLAYER_MAX_HEALTH) * 200)
        pygame.draw.rect(self.screen, RED, (20, 20, 200, 20))
        pygame.draw.rect(self.screen, RED, (20, 20, health_width, 20))
        pygame.draw.rect(self.screen, BLACK, (20, 20, 200, 20), 2)
        
        # Score
        score_text = self.font.render(f'Score: {self.score}', True, BLACK)
        self.screen.blit(score_text, (20, 50))
        
        # Controls hint
        hint_text = self.small_font.render('WASD/Arrows: Move | Left Click: Shoot', True, BLACK)
        self.screen.blit(hint_text, (WIDTH - 400, HEIGHT - 30))
        
        pygame.display.flip()
    
    def game_over(self):
        # Game over screen
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font.render('GAME OVER', True, RED)
        score_text = self.font.render(f'Final Score: {self.score}', True, WHITE)
        restart_text = self.small_font.render('Close window to exit', True, WHITE)
        
        self.screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2 - 50))
        self.screen.blit(score_text, (WIDTH // 2 - 120, HEIGHT // 2))
        self.screen.blit(restart_text, (WIDTH // 2 - 110, HEIGHT // 2 + 50))
        
        pygame.display.flip()
        
        # Wait for exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False

if __name__ == '__main__':
    game = Game()
    game.run()