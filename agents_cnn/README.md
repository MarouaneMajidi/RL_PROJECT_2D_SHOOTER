# CNN-based Agents Implementation

This directory contains **COMPLETELY SEPARATE** CNN-based implementations of Player and DDA agents that use raw RGB images as input instead of manually extracted features.

## Important: Parallel Implementation

- **DO NOT modify** any files in `agents/` (feature-based agents)
- **DO NOT modify** any files in `scripts/` (feature-based training)
- This CNN implementation is **completely independent** and runs in parallel
- Both systems can coexist and be used independently

## Folder Structure

```
agents_cnn/
├── player_cnn/          # CNN-based Player Agent
│   ├── __init__.py
│   ├── config.py        # CNN-specific configuration
│   ├── model.py         # CNN Actor-Critic architecture
│   ├── agent.py         # CNN-based PPO agent
│   ├── memory.py        # Rollout buffer for image states
│   ├── image_preprocessor.py  # Image capture & preprocessing
│   ├── env_wrapper.py   # Environment wrapper for image capture
│   └── metrics_tracker.py     # Metrics tracking (JSON format)
│
└── dda_cnn/             # CNN-based DDA Agent (TODO)
    └── ...
```

## Image Processing

- **Input**: Raw game screenshots from 800×600 arena
- **Processing**: Resize to 128×96 RGB (maintains 4:3 aspect ratio)
- **Frame Stacking**: Stacks 4 consecutive frames for temporal information
- **Final Shape**: (12, 96, 128) = 4 frames × 3 RGB channels, 96 height, 128 width
- **Color**: RGB (NOT grayscale) - color information is preserved

## CNN Architecture

- **Input**: (batch, 12, 96, 128) where 12 = 4 frames × 3 RGB channels
- **Convolutional Layers**: Extract spatial features from images
- **Fully Connected Layers**: Process extracted features
- **Actor Head**: Multi-discrete action space (movement: 5, shoot: 2)
- **Critic Head**: Value estimation

## Training

Training scripts are in `scripts_cnn/`:

- `train_player_cnn.py` - Train CNN-based Player Agent
- `train_dda_cnn.py` - Train CNN-based DDA Agent (TODO)

## Metrics

Metrics are saved in JSON format (same as feature-based agents):
- Location: `checkpoints_cnn/player_cnn/metrics/training_metrics_YYYYMMDD_HHMMSS.json`
- Format: Same structure as existing feature-based agent metrics
- Saved after each episode and at training completion

## Checkpoints

Models are saved in:
- `checkpoints_cnn/player_cnn/best_model.pth` - Best model
- `checkpoints_cnn/player_cnn/checkpoint_*.pth` - Periodic checkpoints
- `checkpoints_cnn/player_cnn/final_model.pth` - Final model

## Usage

See training scripts for command-line arguments. Example:

```bash
# Train CNN-based player agent
python scripts_cnn/train_player_cnn.py --timesteps 1000000
```
