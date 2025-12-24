# CNN-based Agents Implementation Summary

## Overview

A **completely separate** CNN-based implementation of Player and DDA agents that uses raw RGB images as input instead of manually extracted features. This implementation runs in parallel with the existing feature-based agents and does not modify any existing code.

## File Structure

```
agents_cnn/
├── __init__.py
├── README.md                    # Overview and architecture
├── USAGE.md                     # Usage guide and examples
├── IMPLEMENTATION_SUMMARY.md    # This file
│
├── player_cnn/                  # CNN-based Player Agent
│   ├── __init__.py
│   ├── config.py                # CNN Player configuration
│   ├── model.py                 # CNN Actor-Critic (multi-discrete actions)
│   ├── agent.py                 # CNN PPO agent implementation
│   ├── memory.py                # Rollout buffer for image states
│   ├── image_preprocessor.py    # Image capture, resize, frame stacking
│   ├── env_wrapper.py           # Environment wrapper for image capture
│   └── metrics_tracker.py       # Metrics tracking (JSON format)
│
└── dda_cnn/                     # CNN-based DDA Agent
    ├── __init__.py
    ├── config.py                # CNN DDA configuration
    ├── model.py                 # CNN Actor-Critic (discrete actions)
    ├── agent.py                 # CNN DDA PPO agent
    ├── memory.py                # Rollout buffer (reuses player_cnn implementation)
    ├── env_wrapper.py           # DDA environment wrapper for images
    └── metrics_tracker.py       # DDA metrics tracking (JSON format)

scripts_cnn/
├── __init__.py
├── train_player_cnn.py          # Training script for CNN Player Agent
├── train_dda_cnn.py             # Training script for CNN DDA Agent
├── test_player_cnn.py           # Test script for CNN Player Agent
└── test_dda_cnn.py              # Test script for CNN DDA Agent

checkpoints_cnn/
├── player_cnn/
│   ├── best_model.pth
│   ├── checkpoint_*.pth
│   ├── final_model.pth
│   └── metrics/
│       └── training_metrics_YYYYMMDD_HHMMSS.json
│
└── dda_cnn/
    ├── best_model.pth
    ├── checkpoint_*.pth
    ├── final_model.pth
    └── metrics/
        └── dda_training_metrics_YYYYMMDD_HHMMSS.json
```

## Key Components

### 1. Image Preprocessing (`player_cnn/image_preprocessor.py`)

- **Captures**: 800×600 arena region from game screen
- **Resizes**: To 128×96 RGB (maintains 4:3 aspect ratio)
- **Stacks**: 4 consecutive frames for temporal information
- **Output Shape**: (12, 96, 128) where 12 = 4 frames × 3 RGB channels
- **Color**: RGB (NOT grayscale) - preserves color information

### 2. CNN Models

**Player CNN** (`player_cnn/model.py`):
- Input: (batch, 12, 96, 128)
- Architecture: Conv layers → Flatten → Dense → Actor/Critic heads
- Output: Multi-discrete action space (movement: 5, shoot: 2)

**DDA CNN** (`dda_cnn/model.py`):
- Input: (batch, 12, 96, 128)
- Architecture: Conv layers → Flatten → Dense → Actor/Critic heads
- Output: Discrete action space (5 actions: do nothing, easier, harder)

### 3. Environment Wrappers

**Player CNN Wrapper** (`player_cnn/env_wrapper.py`):
- Wraps `ZombieShooterEnv`
- Captures images from `env.screen`
- Provides image-based observations
- Maintains same action/reward interface

**DDA CNN Wrapper** (`dda_cnn/env_wrapper.py`):
- Wraps `DDAEnvironment`
- Captures images from underlying `game_env.screen`
- Provides image-based observations
- Maintains same action/reward interface

### 4. PPO Agents

Both agents implement standard PPO algorithm:
- Clipped surrogate objective
- GAE for advantage estimation
- Explained variance calculation
- Same hyperparameters as feature-based agents

### 5. Metrics Tracking

- **Format**: JSON (same as feature-based agents)
- **Location**: `checkpoints_cnn/{agent}/metrics/`
- **Content**: Episode rewards, lengths, losses, difficulty metrics (DDA), etc.
- **Saving**: After each episode and at training completion
- **Visualization**: Use existing visualization scripts

## Differences from Feature-based Agents

| Aspect | Feature-based | CNN-based |
|--------|---------------|-----------|
| Input | 129-dim feature vector | RGB images (12, 96, 128) |
| Model | MLP (Multi-Layer Perceptron) | CNN (Convolutional Neural Network) |
| Feature Extraction | Manual (hand-crafted) | Learned (automatic) |
| Memory Usage | Lower | Higher (images) |
| Training Speed | Faster | Slower (image processing + CNN) |
| GPU Usage | Lower | Higher (RGB CNN is compute-intensive) |

## Separation from Existing Code

✅ **No modifications** to `agents/` directory  
✅ **No modifications** to `scripts/` directory  
✅ **No modifications** to `checkpoints/` directory  
✅ **No modifications** to `utils/` directory  
✅ **New folders**: `agents_cnn/`, `scripts_cnn/`, `checkpoints_cnn/`  
✅ **Independent**: Both systems can run simultaneously

## Usage Examples

See `USAGE.md` for detailed usage instructions.

### Quick Start

```bash
# Train CNN Player Agent
python scripts_cnn/train_player_cnn.py --timesteps 10000

# Test CNN Player Agent
python scripts_cnn/test_player_cnn.py --model checkpoints_cnn/player_cnn/best_model.pth

# Train CNN DDA Agent
python scripts_cnn/train_dda_cnn.py --player-model checkpoints/player_ppo/best_model.pth --episodes 10
```

## Implementation Status

✅ All core components implemented  
✅ Training scripts with metrics tracking  
✅ Test/evaluation scripts  
✅ Configuration files  
✅ Documentation  
✅ Complete separation from existing code  

## Notes

- CNN agents require rendering even in headless mode (to capture images)
- Frame stacking provides temporal information (motion, trajectories)
- RGB color is preserved (not converted to grayscale)
- Metrics use same JSON format as feature-based agents for consistency
