# Important Note About Rendering and Assets

## Current Situation

**During CNN Training:**
- The CNN captures images from `ZombieShooterEnv.render()`
- `ZombieShooterEnv.render()` uses **simple circles** (`pygame.draw.circle`) - NOT sprite assets
- This means the CNN is training on simple geometric shapes, not the real game graphics

**Real Game (main.py):**
- Uses actual sprite images loaded from `assets/` folder
- Has player sprites, zombie sprites, pickup sprites, etc.
- This is what you see when playing the game manually

## The Problem

If you train the CNN on simple circles but then test it expecting to see the real game with assets, there's a **domain mismatch**:
- Training: Simple circles (ZombieShooterEnv)
- Testing: Real game with sprites (main.py)

The CNN learns to recognize circles, not sprites, so it may not work well with the real game graphics.

## Solutions

### Option 1: Modify ZombieShooterEnv to Use Assets (Recommended)

Modify `agents/ppo_agent/env.py` to use sprite rendering like `main.py` does:

1. Load sprites in `__init__`:
```python
# Load sprites (like main.py does)
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
def load_sprite(filename, scale=1.0):
    path = os.path.join(ASSET_DIR, filename)
    image = pygame.image.load(path).convert_alpha()
    if scale != 1.0:
        new_size = (int(image.get_width() * scale), int(image.get_height() * scale))
        image = pygame.transform.scale(image, new_size)
    return image

player_pistol_img = load_sprite("player pistol.gif", 1.2)
player_machinegun_img = load_sprite("player machinegun.gif", 1.2)
zombie_img = load_sprite("zombie.gif", 1.0)
# etc.
```

2. Modify `render()` method to use sprites instead of circles:
```python
# Instead of:
pygame.draw.circle(self.screen, color, (int(self.player['x']), int(self.player['y'])), 15)

# Use:
img = player_pistol_img if self.player['weapon'] == 'pistol' else player_machinegun_img
rotated_img = pygame.transform.rotate(img, self.player['angle'])
rect = rotated_img.get_rect(center=(int(self.player['x']), int(self.player['y'])))
self.screen.blit(rotated_img, rect)
```

### Option 2: Train with Real Game

Create a new environment wrapper that uses the game loop from `main.py` instead of `ZombieShooterEnv`. This is more complex but ensures the CNN sees exactly what the player sees.

### Option 3: Accept the Mismatch

If the circles are similar enough to sprites in terms of:
- Size
- Color
- Position
- Movement patterns

The CNN might still generalize reasonably well, though performance may be suboptimal.

## Current Implementation

The CNN wrapper has been fixed to ensure rendering happens even in headless mode, so images are captured. However, it still captures the simple circles version from ZombieShooterEnv, not the asset-based version.

## Recommendation

**To get the best results, you should modify `ZombieShooterEnv.render()` to use sprite assets** (Option 1). This ensures:
- CNN trains on real game graphics
- No domain mismatch between training and testing
- Better generalization to the actual game
