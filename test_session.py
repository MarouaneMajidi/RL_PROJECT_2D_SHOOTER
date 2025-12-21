"""
Test Session Manager for DDA Evaluation - Using main.py game logic

Manages test sessions where each player plays 2 matches:
- Match 1: With or without DDA (randomized order)
- Match 2: The other condition

Uses main.py game logic for without_dda matches
Uses play_with_dda.py logic for with_dda matches
Hides wave and DDA information from players.
"""

import os
import sys
import pygame
import math
import random
import json
from datetime import datetime
from typing import Dict
from pygame.locals import *

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from metrics_collector import MetricsCollector

# Import game classes from main.py
# Import as a module to avoid conflicts with main() function
import main as main_module
from main import GameState, Player, Zombie, Bullet, KnifeAttack, Pickup, spawn_zombie

# Import DDA components (optional)
try:
    from dda_agent import DDAAgent, DDAConfig, DDAStateExtractor, DifficultyManager
    DDA_AVAILABLE = True
except ImportError:
    DDA_AVAILABLE = False

# Use main.py's screen and constants
screen = main_module.screen
clock = main_module.clock
ARENA_WIDTH = main_module.ARENA_WIDTH
ARENA_HEIGHT = main_module.ARENA_HEIGHT
SCREEN_WIDTH = main_module.SCREEN_WIDTH
SCREEN_HEIGHT = main_module.SCREEN_HEIGHT
ARENA_X_OFFSET = main_module.ARENA_X_OFFSET
ARENA_Y_OFFSET = main_module.ARENA_Y_OFFSET
FPS = main_module.FPS
BG_COLOR = main_module.BG_COLOR

# Base difficulty parameters for DDA
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
YELLOW = (255, 255, 0)


def draw_ui_hidden_wave(screen, player, zombies_killed):
    """Draw UI - HIDES wave information for blind testing."""
    font = pygame.font.SysFont(None, 36)
    
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    
    # HIDDEN: Wave information is NOT displayed
    
    kills_text = font.render(f"Zombies Killed: {zombies_killed}", True, WHITE)
    screen.blit(kills_text, (10, 50))
    
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


def wait_for_key(screen, message="Press ENTER to continue..."):
    """Wait for user to press ENTER key, displaying message on screen."""
    font = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 36)
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_RETURN or event.key == K_SPACE:
                    waiting = False
        
        # Draw message
        screen.fill(BLACK)
        text = font.render(message, True, WHITE)
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 50))
        
        inst_text = font_small.render("Press ENTER or SPACE to continue", True, YELLOW)
        screen.blit(inst_text, (SCREEN_WIDTH//2 - inst_text.get_width()//2, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip()
        clock.tick(FPS)


def ask_question(screen, question, scale_min=1, scale_max=5):
    """Ask a question and return the answer."""
    font = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 36)
    
    answer = None
    selected = 3 if scale_min == 1 and scale_max == 5 else None
    
    while answer is None:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_LEFT and selected is not None:
                    selected = max(scale_min, selected - 1)
                elif event.key == K_RIGHT and selected is not None:
                    selected = min(scale_max, selected + 1)
                elif event.key == K_RETURN or event.key == K_SPACE:
                    answer = selected
        
        screen.fill(BLACK)
        
        # Question
        question_surface = font.render(question, True, WHITE)
        screen.blit(question_surface, (SCREEN_WIDTH//2 - question_surface.get_width()//2, 200))
        
        # Scale
        if scale_min == 1 and scale_max == 5:
            labels = ["1", "2", "3", "4", "5"]
            if "Difficulté" in question or "difficulté" in question:
                label_texts = ["Trop facile", "", "Équilibré", "", "Trop difficile"]
            else:
                label_texts = ["Pas du tout", "", "", "", "Beaucoup"]
        else:
            labels = []
            label_texts = []
        
        for i, label in enumerate(labels):
            x = SCREEN_WIDTH//2 - 200 + i * 100
            y = 350
            color = YELLOW if selected == int(label) else WHITE
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (x - label_surface.get_width()//2, y))
            
            if i < len(label_texts) and label_texts[i]:
                text_surface = font_small.render(label_texts[i], True, WHITE)
                screen.blit(text_surface, (x - text_surface.get_width()//2, y + 40))
        
        # Instructions
        inst_text = font_small.render("Flèches gauche/droite pour choisir, Entrée pour confirmer", True, WHITE)
        screen.blit(inst_text, (SCREEN_WIDTH//2 - inst_text.get_width()//2, 500))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    return answer


def ask_preference(screen):
    """Ask preference question."""
    font = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 36)
    
    answer = None
    selected = 0  # 0 = Match 1, 1 = Match 2, 2 = Aucune préférence
    
    while answer is None:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selected = max(0, selected - 1)
                elif event.key == K_DOWN:
                    selected = min(2, selected + 1)
                elif event.key == K_RETURN or event.key == K_SPACE:
                    if selected == 0:
                        answer = "match1"
                    elif selected == 1:
                        answer = "match2"
                    else:
                        answer = "no_preference"
        
        screen.fill(BLACK)
        
        question = "Quelle version avez-vous préférée ?"
        question_surface = font.render(question, True, WHITE)
        screen.blit(question_surface, (SCREEN_WIDTH//2 - question_surface.get_width()//2, 200))
        
        options = ["Le premier match", "Le deuxième match", "Aucune préférence"]
        for i, option in enumerate(options):
            y = 300 + i * 60
            color = YELLOW if selected == i else WHITE
            option_surface = font_small.render(option, True, color)
            screen.blit(option_surface, (SCREEN_WIDTH//2 - option_surface.get_width()//2, y))
        
        inst_text = font_small.render("Flèches haut/bas pour choisir, Entrée pour confirmer", True, WHITE)
        screen.blit(inst_text, (SCREEN_WIDTH//2 - inst_text.get_width()//2, 500))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    return answer


def run_match_without_dda(player_id, match_number, collector):
    """Run match without DDA using main.py GameState logic."""
    game = GameState()
    game.resetGame()
    
    # Track events for metrics
    frame_damage_taken = 0
    frame_damage_dealt = 0
    frame_zombie_killed = False
    frame_shot_fired = False
    frame_shot_hit = False
    frame_pickup_collected = False
    last_metrics_time = 0
    last_health = 100
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        
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
                    frame_shot_fired = True
                else:
                    game.knife_attacks.append(new_bullet)
                    frame_shot_fired = True
            
            # Track damage taken
            current_damage_taken = max(0, last_health - game.player.health)
            if current_damage_taken > 0:
                frame_damage_taken = current_damage_taken
            last_health = game.player.health
            
            # Update pickups and check collisions
            pickups_to_remove = []
            for pickup in game.pickups:
                if pickup.update():
                    pickups_to_remove.append(pickup)
                elif pickup.check_collision(game.player.x, game.player.y):
                    if pickup.pickup_type == "health":
                        game.player.add_health(main_module.HEALTH_PICKUP_AMOUNT)
                    elif pickup.pickup_type == "machinegun":
                        game.player.pickup_machinegun(main_module.MACHINEGUN_PICKUP_AMMO)
                    pickups_to_remove.append(pickup)
                    frame_pickup_collected = True
            
            for pickup in pickups_to_remove:
                if pickup in game.pickups:
                    game.pickups.remove(pickup)
            
            # Update bullets
            bullets_to_remove = []
            zombies_to_remove = []
            for bullet in game.bullets:
                if bullet.update():
                    bullets_to_remove.append(bullet)
                else:
                    for zombie in game.zombies:
                        if bullet.check_collision(zombie):
                            damage = bullet.damage
                            frame_damage_dealt += damage
                            if zombie.take_damage(damage):
                                zombies_to_remove.append(zombie)
                                game.player.score += 10 if zombie.zombie_type == "normal" else 20
                                game.zombies_killed += 1
                                frame_zombie_killed = True
                                frame_shot_hit = True
                                drop_chance = random.random()
                                if drop_chance < main_module.HEALTH_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "health"))
                                elif drop_chance < main_module.HEALTH_PICKUP_DROP_PROBABILITY + main_module.MACHINEGUN_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "machinegun"))
                            bullets_to_remove.append(bullet)
                            break
            
            for bullet in bullets_to_remove:
                if bullet in game.bullets:
                    game.bullets.remove(bullet)
            
            # Update knife attacks
            knife_attacks_to_remove = []
            for knife in game.knife_attacks:
                if knife.update():
                    knife_attacks_to_remove.append(knife)
                else:
                    for zombie in game.zombies:
                        if knife.check_collision(zombie):
                            damage = knife.damage
                            frame_damage_dealt += damage
                            if zombie.take_damage(damage):
                                zombies_to_remove.append(zombie)
                                game.player.score += 10 if zombie.zombie_type == "normal" else 20
                                game.zombies_killed += 1
                                frame_zombie_killed = True
                                drop_chance = random.random()
                                if drop_chance < main_module.HEALTH_PICKUP_DROP_PROBABILITY:
                                    game.pickups.append(Pickup(zombie.x, zombie.y, "health"))
                                elif drop_chance < main_module.HEALTH_PICKUP_DROP_PROBABILITY + main_module.MACHINEGUN_PICKUP_DROP_PROBABILITY:
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
                            running = False
            
            # Record metrics every second
            current_time = pygame.time.get_ticks() / 1000.0
            if current_time - last_metrics_time >= 1.0:
                game_state = {
                    'player_health': game.player.health,
                    'zombies_alive': len(game.zombies),
                    'player_x': game.player.x,
                    'player_y': game.player.y,
                    'score': game.player.score,
                    'damage_taken': frame_damage_taken,
                    'damage_dealt': frame_damage_dealt,
                    'zombie_killed': frame_zombie_killed,
                    'shot_fired': frame_shot_fired,
                    'shot_hit': frame_shot_hit,
                    'pickup_collected': frame_pickup_collected,
                }
                collector.record_frame(game_state)
                
                # Reset frame events
                frame_damage_taken = 0
                frame_damage_dealt = 0
                frame_zombie_killed = False
                frame_shot_fired = False
                frame_shot_hit = False
                frame_pickup_collected = False
                last_metrics_time = current_time
        
        # Draw everything (same as main.py)
        screen.fill(BG_COLOR)
        arena_rect = pygame.Rect(ARENA_X_OFFSET, ARENA_Y_OFFSET, ARENA_WIDTH, ARENA_HEIGHT)
        if main_module.background_img:
            screen.blit(main_module.background_img, (ARENA_X_OFFSET, ARENA_Y_OFFSET))
        else:
            pygame.draw.rect(screen, (80, 80, 80), arena_rect)
        
        # Draw glass borders
        glass_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        left_border = pygame.Rect(ARENA_X_OFFSET - main_module.GLASS_BORDER_THICKNESS, ARENA_Y_OFFSET,
                                 main_module.GLASS_BORDER_THICKNESS, ARENA_HEIGHT)
        right_border = pygame.Rect(ARENA_X_OFFSET + ARENA_WIDTH, ARENA_Y_OFFSET,
                                  main_module.GLASS_BORDER_THICKNESS, ARENA_HEIGHT)
        pygame.draw.rect(glass_surface, main_module.GLASS_COLOR, left_border)
        pygame.draw.rect(glass_surface, main_module.GLASS_COLOR, right_border)
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
        
        # Draw UI (HIDES wave information)
        draw_ui_hidden_wave(screen, game.player, game.zombies_killed)
        
        if game.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            font_large = pygame.font.SysFont(None, 72)
            screen.blit(font_large.render("GAME OVER", True, RED), (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 50))
            screen.blit(pygame.font.SysFont(None, 36).render("Press ESC to continue", True, WHITE),
                       (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip()
        clock.tick(FPS)


def run_match_with_dda(player_id, match_number, collector, dda_model_path):
    """Run match with DDA using play_with_dda.py logic."""
    # Import play_with_dda classes
    import play_with_dda as pwd
    
    # Load DDA agent
    dda_config = DDAConfig()
    dda_agent = DDAAgent(dda_config)
    
    try:
        dda_agent.load(dda_model_path)
        use_dda = True
    except FileNotFoundError:
        print(f"Warning: DDA model not found at {dda_model_path}")
        use_dda = False
    
    # Initialize DDA components
    state_extractor = DDAStateExtractor(window_size=600, fps=60)
    difficulty_manager = DifficultyManager(BASE_PARAMS)
    
    # Game state (using play_with_dda.py classes)
    params = difficulty_manager.get_current_params()
    player = pwd.Player(params)
    zombies = []
    bullets = []
    pickups = []
    
    spawn_timer = 0
    zombies_killed = 0
    dda_action_timer = 0
    dda_action_interval = 300  # 5 seconds
    last_dda_action = "Initializing..."
    
    # Metrics tracking
    last_health = 100
    frame_damage_taken = 0
    frame_damage_dealt = 0
    frame_zombie_killed = False
    frame_shot_fired = False
    frame_shot_hit = False
    frame_pickup_collected = False
    last_metrics_time = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        
        if player.health > 0:
            # Input
            keys = pygame.key.get_pressed()
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            # Update player
            new_bullet = player.update(keys, mouse_pos, mouse_buttons)
            if new_bullet:
                bullets.append(new_bullet)
                frame_shot_fired = True
            
            # Track metrics
            damage_taken = max(0, last_health - player.health)
            last_health = player.health
            if damage_taken > 0:
                frame_damage_taken = damage_taken
            
            state_extractor.update({
                'player_health': player.health,
                'damage_taken': frame_damage_taken,
                'damage_dealt': frame_damage_dealt,
                'zombie_killed': frame_zombie_killed,
                'shot_fired': frame_shot_fired,
                'shot_hit': frame_shot_hit,
                'pickup_collected': frame_pickup_collected,
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
                player.params = params  # Update player params
                
                # Update zombie speeds
                for z in zombies:
                    z.speed = params['ZOMBIE_SPEED']
                
                action_names = ["Do Nothing", "Slightly Easier", "Much Easier", "Slightly Harder", "Much Harder"]
                last_dda_action = action_names[action]
            
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
                    frame_pickup_collected = True
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
                            frame_damage_dealt += bullet.damage
                            if zombie.health <= 0:
                                zombies.remove(zombie)
                                zombies_killed += 1
                                player.score += 10
                                frame_zombie_killed = True
                                frame_shot_hit = True
                                # Drop pickup
                                if random.random() < params['HEALTH_PICKUP_DROP_PROBABILITY']:
                                    pickups.append(pwd.Pickup(zombie.x, zombie.y, "health"))
                                elif random.random() < params['MACHINEGUN_PICKUP_DROP_PROBABILITY']:
                                    pickups.append(pwd.Pickup(zombie.x, zombie.y, "machinegun"))
                            bullets_to_remove.append(bullet)
                            break
            for b in bullets_to_remove:
                if b in bullets:
                    bullets.remove(b)
            
            # Spawn zombies
            spawn_timer += 1
            if spawn_timer >= params['SPAWN_RATE']:
                zombies.append(pwd.spawn_zombie(params))
                spawn_timer = 0
            
            # Update zombies
            for zombie in zombies:
                zombie.update(player.x, player.y)
                if math.hypot(zombie.x - player.x, zombie.y - player.y) < 30:
                    if zombie.attack_cooldown == 0:
                        player.health -= zombie.damage
                        zombie.attack_cooldown = int(params['ZOMBIE_ATTACK_COOLDOWN'])
                        if player.health <= 0:
                            running = False
            
            # Record metrics every second
            current_time = pygame.time.get_ticks() / 1000.0
            if current_time - last_metrics_time >= 1.0:
                game_state = {
                    'player_health': player.health,
                    'zombies_alive': len(zombies),
                    'player_x': player.x,
                    'player_y': player.y,
                    'score': player.score,
                    'damage_taken': frame_damage_taken,
                    'damage_dealt': frame_damage_dealt,
                    'zombie_killed': frame_zombie_killed,
                    'shot_fired': frame_shot_fired,
                    'shot_hit': frame_shot_hit,
                    'pickup_collected': frame_pickup_collected,
                    'dda_action': last_dda_action,
                    'difficulty_params': params.copy(),
                }
                collector.record_frame(game_state)
                
                # Reset frame events
                frame_damage_taken = 0
                frame_damage_dealt = 0
                frame_zombie_killed = False
                frame_shot_fired = False
                frame_shot_hit = False
                frame_pickup_collected = False
                last_metrics_time = current_time
        
        # Draw (same as play_with_dda.py)
        screen.fill(BG_COLOR)
        screen.blit(pwd.background_img, (ARENA_X_OFFSET, ARENA_Y_OFFSET))
        
        player.draw(screen)
        for zombie in zombies:
            zombie.draw(screen)
        for bullet in bullets:
            bullet.draw(screen)
        for pickup in pickups:
            pickup.draw(screen)
        
        # Draw UI (HIDES wave and DDA information)
        draw_ui_hidden_wave(screen, player, zombies_killed)
        
        if player.health <= 0:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            font_large = pygame.font.SysFont(None, 72)
            screen.blit(font_large.render("GAME OVER", True, RED), (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 50))
            screen.blit(pygame.font.SysFont(None, 36).render("Press ESC to continue", True, WHITE),
                       (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip()
        clock.tick(FPS)


def run_match(player_id, match_number, condition, dda_model_path=None):
    """Run a single match."""
    # Initialize metrics collector
    collector = MetricsCollector(player_id, match_number, condition)
    collector.start_match()
    
    if condition == "with_dda":
        run_match_with_dda(player_id, match_number, collector, dda_model_path)
    else:
        run_match_without_dda(player_id, match_number, collector)
    
    # End match
    collector.end_match()
    consolidated_metrics, consolidated_frames = collector.save_to_consolidated()
    
    return consolidated_metrics, consolidated_frames


def get_next_player_id(data_dir: str = "evaluation_data") -> int:
    """Get the next available player ID."""
    os.makedirs(data_dir, exist_ok=True)
    counter_file = os.path.join(data_dir, "player_counter.json")
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            data = json.load(f)
            next_id = data.get('next_id', 1)
    else:
        next_id = 1
    
    with open(counter_file, 'w') as f:
        json.dump({'next_id': next_id + 1}, f, indent=2)
    
    return next_id


def append_to_consolidated_file(data_dir: str, filename: str, data: Dict):
    """Append data to a consolidated JSON file."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = []
    
    all_data.append(data)
    
    with open(filepath, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    return filepath


def main():
    """Main test session manager."""
    print("=" * 60)
    print("DDA Evaluation Test Session")
    print("=" * 60)
    
    # Get next player ID automatically
    player_id = get_next_player_id()
    print(f"\nPlayer ID: {player_id} (assigned automatically)")
    
    # Randomize order
    order = random.choice(["dda_first", "dda_second"])
    if order == "dda_first":
        match1_condition = "with_dda"
        match2_condition = "without_dda"
    else:
        match1_condition = "without_dda"
        match2_condition = "with_dda"
    
    # Don't show condition to player - keep it blind
    print(f"Match 1: {match1_condition}")  # Only in console for debugging
    print(f"Match 2: {match2_condition}")  # Only in console for debugging
    print("\nStarting Match 1...")
    
    # Show message on screen (don't reveal condition)
    wait_for_key(screen, "Match 1 - Press ENTER to start")
    
    # Run Match 1
    dda_model_path = "checkpoints/dda/best_dda_model.pth" if match1_condition == "with_dda" else None
    consolidated_metrics1, consolidated_frames1 = run_match(player_id, 1, match1_condition, dda_model_path)
    print(f"\nMatch 1 completed! Data saved to consolidated files.")
    
    # Ask questions after Match 1
    print("\nPlease answer the following questions:")
    wait_for_key(screen, "Please answer the questions...")
    
    enjoyment1 = ask_question(screen, "À quel point avez-vous apprécié ce match ?", 1, 5)
    difficulty1 = ask_question(screen, "Le jeu était-il trop facile, trop difficile, ou équilibré ?", 1, 5)
    
    # Save answers
    append_to_consolidated_file("evaluation_data", "all_answers.json", {
        'player_id': player_id,
        'match_number': 1,
        'condition': match1_condition,
        'enjoyment': enjoyment1,
        'difficulty': difficulty1,
        'timestamp': datetime.now().isoformat(),
    })
    
    print(f"\nMatch 1 questions answered!")
    wait_for_key(screen, "Match 2 - Press ENTER to start")
    
    # Run Match 2
    dda_model_path = "checkpoints/dda/best_dda_model.pth" if match2_condition == "with_dda" else None
    consolidated_metrics2, consolidated_frames2 = run_match(player_id, 2, match2_condition, dda_model_path)
    print(f"\nMatch 2 completed! Data saved to consolidated files.")
    
    # Ask questions after Match 2
    print("\nPlease answer the following questions:")
    wait_for_key(screen, "Please answer the questions...")
    
    enjoyment2 = ask_question(screen, "À quel point avez-vous apprécié ce match ?", 1, 5)
    difficulty2 = ask_question(screen, "Le jeu était-il trop facile, trop difficile, ou équilibré ?", 1, 5)
    
    # Save answers
    append_to_consolidated_file("evaluation_data", "all_answers.json", {
        'player_id': player_id,
        'match_number': 2,
        'condition': match2_condition,
        'enjoyment': enjoyment2,
        'difficulty': difficulty2,
        'timestamp': datetime.now().isoformat(),
    })
    
    # Ask preference question
    print("\nFinal question:")
    wait_for_key(screen, "Final question...")
    
    preference = ask_preference(screen)
    
    # Map preference to condition
    if preference == "match1":
        preferred_condition = match1_condition
    elif preference == "match2":
        preferred_condition = match2_condition
    else:
        preferred_condition = "no_preference"
    
    # Save preference
    append_to_consolidated_file("evaluation_data", "all_preferences.json", {
        'player_id': player_id,
        'match1_condition': match1_condition,
        'match2_condition': match2_condition,
        'preference': preference,
        'preferred_condition': preferred_condition,
        'timestamp': datetime.now().isoformat(),
    })
    
    print(f"\n" + "=" * 60)
    print(f"Test session completed!")
    print(f"=" * 60)
    print(f"Player ID: {player_id}")
    print(f"Match 1: {match1_condition}")
    print(f"Match 2: {match2_condition}")
    print(f"Preference: {preferred_condition}")
    print(f"\nAll data saved to evaluation_data/")
    print(f"- all_metrics.json (all metrics from all players)")
    print(f"- all_frames.json (all frame data from all players)")
    print(f"- all_answers.json (all answers from all players)")
    print(f"- all_preferences.json (all preferences from all players)")
    print(f"\nNext player ID will be: {player_id + 1}")
    print(f"=" * 60)
    
    pygame.quit()


if __name__ == "__main__":
    main()

