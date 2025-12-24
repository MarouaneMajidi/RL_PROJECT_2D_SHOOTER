"""
Visualization script to show all pickup types in the game.
Run this to see what each pickup looks like.
"""

import pygame
import sys
import os

# Initialize pygame
pygame.init()

# Screen setup
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pickup Types Visualization")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (150, 150, 150)
DARK_YELLOW = (200, 200, 0)
DARK = (100, 100, 0)
BLACK = (0, 0, 0)

# Font for labels
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 32)
font_small = pygame.font.Font(None, 24)

def draw_health_pickup(screen, x, y):
    """Draw health pickup (red cross)"""
    # Red cross
    pygame.draw.rect(screen, RED, (x - 15, y - 3, 30, 6))
    pygame.draw.rect(screen, RED, (x - 3, y - 15, 6, 30))
    # White circle border
    pygame.draw.circle(screen, WHITE, (x, y), 18, 2)

def draw_machinegun_pickup(screen, x, y):
    """Draw machinegun pickup (gray box with yellow indicator)"""
    # Gray box (gun shape)
    pygame.draw.rect(screen, GRAY, (x - 12, y - 4, 24, 8))
    pygame.draw.rect(screen, LIGHT_GRAY, (x - 8, y - 2, 16, 4))
    # Yellow indicator dot
    pygame.draw.circle(screen, YELLOW, (x + 10, y), 3)
    # White circle border
    pygame.draw.circle(screen, WHITE, (x, y), 18, 2)

def draw_ammo_pickup(screen, x, y):
    """Draw ammo pickup (yellow box with bullets)"""
    # Yellow box
    pygame.draw.rect(screen, YELLOW, (x - 10, y - 5, 20, 10))
    pygame.draw.rect(screen, DARK_YELLOW, (x - 8, y - 3, 16, 6))
    # Bullet indicators (3 dots)
    for i in range(3):
        pygame.draw.circle(screen, DARK, (x - 5 + i * 5, y), 2)
    # White circle border
    pygame.draw.circle(screen, WHITE, (x, y), 18, 2)

def main():
    """Main visualization loop"""
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw title
        title = font_large.render("Pickup Types in Zombie Shooter", True, WHITE)
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 30))
        
        # Calculate positions for three pickups side by side
        center_y = SCREEN_HEIGHT // 2 + 50
        spacing = SCREEN_WIDTH // 4
        start_x = SCREEN_WIDTH // 4
        
        # 1. Health Pickup
        health_x = start_x
        health_label = font_medium.render("HEALTH", True, WHITE)
        screen.blit(health_label, (health_x - health_label.get_width() // 2, center_y - 100))
        health_desc = font_small.render("Restores 25 HP", True, WHITE)
        screen.blit(health_desc, (health_x - health_desc.get_width() // 2, center_y - 70))
        draw_health_pickup(screen, health_x, center_y)
        
        # 2. Machinegun Pickup
        mg_x = start_x + spacing
        mg_label = font_medium.render("MACHINEGUN", True, WHITE)
        screen.blit(mg_label, (mg_x - mg_label.get_width() // 2, center_y - 100))
        mg_desc = font_small.render("Gives weapon + 50 ammo", True, WHITE)
        screen.blit(mg_desc, (mg_x - mg_desc.get_width() // 2, center_y - 70))
        draw_machinegun_pickup(screen, mg_x, center_y)
        
        # 3. Ammo Pickup
        ammo_x = start_x + spacing * 2
        ammo_label = font_medium.render("AMMO", True, WHITE)
        screen.blit(ammo_label, (ammo_x - ammo_label.get_width() // 2, center_y - 100))
        ammo_desc = font_small.render("Gives 30 ammo (need MG)", True, WHITE)
        screen.blit(ammo_desc, (ammo_x - ammo_desc.get_width() // 2, center_y - 70))
        draw_ammo_pickup(screen, ammo_x, center_y)
        
        # Instructions
        instructions = font_small.render("Press ESC or close window to exit", True, WHITE)
        screen.blit(instructions, (SCREEN_WIDTH // 2 - instructions.get_width() // 2, SCREEN_HEIGHT - 40))
        
        # Note about ammo
        note = font_small.render("Note: Ammo pickup is ONLY useful if you already have the machinegun!", True, YELLOW)
        screen.blit(note, (SCREEN_WIDTH // 2 - note.get_width() // 2, SCREEN_HEIGHT - 70))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
