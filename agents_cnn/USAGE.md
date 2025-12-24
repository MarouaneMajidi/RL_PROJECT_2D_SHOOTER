# CNN-based Agents Usage Guide

This guide explains how to use the CNN-based agents that process raw RGB images.

## Overview

The CNN-based agents are **completely separate** from the feature-based agents:
- **Feature-based agents**: Located in `agents/` - use manually extracted 129-dim feature vectors
- **CNN-based agents**: Located in `agents_cnn/` - use raw RGB images (128×96, 4 stacked frames)

Both systems can run independently and coexist.

## Image Processing

- **Input Resolution**: 800×600 arena (original game resolution)
- **Resized Resolution**: 128×96 RGB (maintains 4:3 aspect ratio)
- **Frame Stacking**: 4 consecutive frames stacked together
- **Final Input Shape**: (12, 96, 128) = 4 frames × 3 RGB channels, 96 height, 128 width
- **Color**: RGB (NOT grayscale) - color information is preserved for the CNN

## Training CNN-based Player Agent

```bash
# Train with default settings (1,000,000 timesteps)
python scripts_cnn/train_player_cnn.py

# Train for specific number of timesteps
python scripts_cnn/train_player_cnn.py --timesteps 500000

# Train with custom hyperparameters
python scripts_cnn/train_player_cnn.py --timesteps 1000000 --lr 3e-4 --batch-size 64

# Continue training from checkpoint
python scripts_cnn/train_player_cnn.py --checkpoint checkpoints_cnn/player_cnn/checkpoint_500000.pth
```

## Training CNN-based DDA Agent

```bash
# Train with default settings (requires trained player agent)
python scripts_cnn/train_dda_cnn.py --player-model checkpoints/player_ppo/best_model.pth

# Train for specific number of episodes
python scripts_cnn/train_dda_cnn.py --player-model checkpoints/player_ppo/best_model.pth --episodes 50

# Continue training from checkpoint
python scripts_cnn/train_dda_cnn.py --player-model checkpoints/player_ppo/best_model.pth \
    --checkpoint checkpoints_cnn/dda_cnn/checkpoint_250000.pth
```

## Testing CNN Agents

### Test Player Agent
```bash
python scripts_cnn/test_player_cnn.py --model checkpoints_cnn/player_cnn/best_model.pth --episodes 10
```

### Test DDA Agent
```bash
python scripts_cnn/test_dda_cnn.py --dda-model checkpoints_cnn/dda_cnn/best_model.pth \
    --player-model checkpoints/player_ppo/best_model.pth --episodes 5
```

## Metrics

Metrics are saved in JSON format (same as feature-based agents):

**Player Agent Metrics:**
- Location: `checkpoints_cnn/player_cnn/metrics/training_metrics_YYYYMMDD_HHMMSS.json`
- Use existing visualization script: `scripts/visualize_training_metrics.py`

**DDA Agent Metrics:**
- Location: `checkpoints_cnn/dda_cnn/metrics/dda_training_metrics_YYYYMMDD_HHMMSS.json`
- Use existing visualization script: `scripts/visualize_dda_training_metrics.py`

## Visualization

You can use the existing visualization scripts (they work with any JSON metrics file):

```bash
# Visualize player CNN metrics
python scripts/visualize_training_metrics.py checkpoints_cnn/player_cnn/metrics/training_metrics_*.json

# Visualize DDA CNN metrics
python scripts/visualize_dda_training_metrics.py checkpoints_cnn/dda_cnn/metrics/dda_training_metrics_*.json
```

## Checkpoints

Models are saved in separate directories:
- **Player CNN**: `checkpoints_cnn/player_cnn/`
- **DDA CNN**: `checkpoints_cnn/dda_cnn/`

This keeps them completely separate from feature-based checkpoints in `checkpoints/`.

## Key Differences from Feature-based Agents

1. **Input**: RGB images instead of feature vectors
2. **Model Architecture**: CNN instead of MLP
3. **Memory Usage**: Higher (images are larger than feature vectors)
4. **Training Speed**: Potentially slower (image processing + CNN forward/backward passes)
5. **GPU Usage**: More GPU-intensive (RGB CNN requires more computation)

## Notes

- The CNN agents use the same action spaces and reward functions as feature-based agents
- Both systems can be trained and tested independently
- CNN agents require the game to render (even in headless mode) to capture screenshots
- Frame stacking provides temporal information (motion, trajectories)
