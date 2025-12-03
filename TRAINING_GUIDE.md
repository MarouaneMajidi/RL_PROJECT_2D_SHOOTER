# PPO Zombie Shooter - Training and Usage Guide

## 🎮 Project Overview

This project implements a complete **Proximal Policy Optimization (PPO)** reinforcement learning agent that learns to play a 2D top-down zombie shooter game built with Pygame.

### Features

- ✅ **From-scratch PPO implementation** (no Stable Baselines)
- ✅ **Actor-Critic architecture** with PyTorch
- ✅ **Generalized Advantage Estimation (GAE)**
- ✅ **Clipped surrogate objective**
- ✅ **Modular, production-grade code**
- ✅ **Dual modes**: Manual (human) and Agent (AI)
- ✅ **Comprehensive reward shaping**
- ✅ **Model checkpointing and logging**

---

## 📁 Project Structure

```
RL_PROJECT_2D_SHOOTER/
├── ppo_agent/              # PPO implementation package
│   ├── __init__.py
│   ├── agent.py           # PPO algorithm with clipped objective
│   ├── model.py           # Actor-Critic neural network
│   ├── memory.py          # Rollout buffer with GAE
│   ├── env.py             # Gym-style environment wrapper
│   └── config.py          # Hyperparameters
├── main.py                # Game with manual/agent modes
├── train_ppo.py           # Training script
├── test_agent.py          # Testing script
├── assets/                # Game sprites
├── checkpoints/           # Saved models (created during training)
├── logs/                  # Training logs (created during training)
└── pyproject.toml         # Dependencies
```

---

## 🚀 Installation

### 1. Clone and Navigate

```bash
cd /path/to/RL_PROJECT_2D_SHOOTER
```

### 2. Install Dependencies

**Option A: Using uv (recommended)**
```bash
uv sync
```

**Option B: Using pip**
```bash
pip install pygame torch numpy
```

**For CUDA support (GPU training):**
```bash
# Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Quick Start

### Play Manually (Human Control)

```bash
python main.py --mode manual
```

**Controls:**
- `WASD` or Arrow Keys: Move
- Mouse: Aim
- Left Click: Shoot
- `1`: Pistol
- `2`: Machine Gun (when available)
- `3`: Knife

---

## 🤖 Training the PPO Agent

### Basic Training

```bash
python train_ppo.py
```

### Advanced Training Options

```bash
# Train for 2 million timesteps with custom hyperparameters
python train_ppo.py --timesteps 2000000 --lr 0.0003 --batch-size 64

# Train with rendering (slower, for visualization)
python train_ppo.py --render

# Continue from checkpoint
python train_ppo.py --checkpoint checkpoints/checkpoint_500000.pth

# Use CPU instead of GPU
python train_ppo.py --device cpu
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 1,000,000 | Total training timesteps |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 64 | Mini-batch size |
| `--n-steps` | 2048 | Rollout buffer size |
| `--device` | cuda | Training device (cuda/cpu) |
| `--render` | False | Render during training |
| `--checkpoint` | None | Path to checkpoint to resume |

### What to Expect During Training

```
Using device: cuda

Creating environment...
Creating PPO agent...

Starting training...
Total timesteps: 1,000,000
Rollout steps: 2048
Batch size: 64
Epochs per update: 10

======================================================================
Timestep: 2,048 | Episode: 1
Episode Reward: 45.20 | Length: 512
Zombies Killed: 3

--- Update 1 ---
Timesteps: 2,048 / 1,000,000
Policy Loss: 0.0234
Value Loss: 0.4521
Entropy: -1.7821
Clip Fraction: 0.1234
```

Training typically takes **2-4 hours** on a GPU for 1 million timesteps.

---

## 🧪 Testing the Trained Agent

### Watch the Agent Play

```bash
python test_agent.py --model checkpoints/best_model.pth
```

### Test Multiple Episodes

```bash
# Run 20 test episodes
python test_agent.py --model checkpoints/best_model.pth --episodes 20

# Run without rendering (for statistics)
python test_agent.py --model checkpoints/best_model.pth --episodes 100 --no-render

# Test with stochastic policy (adds randomness)
python test_agent.py --model checkpoints/best_model.pth --stochastic
```

### Example Output

```
========================================================
Episode 1/10
========================================================
  Steps: 100, Reward: 12.50, Kills: 2, Health: 85
  Steps: 200, Reward: 28.30, Kills: 5, Health: 70

Episode 1 finished:
  Total Reward: 156.20
  Episode Length: 523
  Zombies Killed: 12
  Final Health: 45
  Result: DIED

========================================================
TEST SUMMARY (10 episodes)
========================================================
Average Reward: 142.35 ± 28.47
Average Length: 485.3 ± 112.8
Average Kills: 11.2 ± 3.4
Best Reward: 198.50
Worst Reward: 98.20
```

---

## 🎮 Watch the Agent in the Game

Run the game in agent mode to see the trained PPO agent play:

```bash
python main.py --mode agent --model checkpoints/best_model.pth
```

**Controls:**
- `ESC`: Quit
- `R`: Restart episode

---

## ⚙️ PPO Algorithm Details

### State Space (98-dimensional vector)

**Player State (8 values):**
- Position (x, y)
- Velocity (vx, vy) 
- Health
- Shoot cooldown
- Aim angle
- Zombie count

**Zombie State (6 values per zombie, max 15 zombies = 90 values):**
- Position (x, y)
- Health
- Type (normal/strong)
- Distance to player
- Angle to player

### Action Space (Discrete, 6 actions)

- 0: Move Up
- 1: Move Down
- 2: Move Left
- 3: Move Right
- 4: Shoot
- 5: Idle

### Reward Function

| Event | Reward |
|-------|--------|
| Kill zombie | +10.0 |
| Survive (per step) | +0.1 |
| Take damage | -5.0 per 10 HP |
| Death | -20.0 |
| Idle too long | -0.01 |
| Move closer to zombie | +0.01 |

### PPO Hyperparameters

```python
Learning Rate:      3e-4
Discount (γ):       0.99
GAE Lambda (λ):     0.95
Clip Epsilon (ε):   0.2
Value Loss Coef:    0.5
Entropy Coef:       0.01
Gradient Clip:      0.5
Rollout Steps:      2048
Batch Size:         64
Update Epochs:      10
```

### Network Architecture

```
Actor-Critic Network:
├── Shared Feature Extractor
│   ├── Linear(98 → 256) + ReLU
│   └── Linear(256 → 256) + ReLU
├── Actor Head (Policy)
│   ├── Linear(256 → 128) + ReLU
│   └── Linear(128 → 6)  # Action logits
└── Critic Head (Value)
    ├── Linear(256 → 128) + ReLU
    └── Linear(128 → 1)   # State value
```

---

## 📊 Monitoring Training

### Log Files

Training logs are saved to `logs/training_YYYYMMDD_HHMMSS.log` with:
- Episode rewards
- Episode lengths
- Zombies killed
- Policy/value/entropy losses
- Clip fractions
- Approximate KL divergence

### Checkpoints

Models are saved to `checkpoints/`:
- `best_model.pth` - Best performing model
- `final_model.pth` - Final trained model
- `checkpoint_N.pth` - Periodic checkpoints every 50k steps

---

## 🔧 Hyperparameter Tuning

Edit `ppo_agent/config.py` to customize:

```python
# Example modifications
config = PPOConfig()
config.learning_rate = 1e-4        # Lower LR for stability
config.clip_epsilon = 0.1          # More conservative updates
config.n_steps = 4096              # Longer rollouts
config.entropy_coef = 0.02         # More exploration
config.max_zombies = 20            # Track more zombies
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_ppo.py --batch-size 32

# Use CPU
python train_ppo.py --device cpu
```

### Poor Performance

- **Increase training time**: `--timesteps 2000000`
- **Adjust learning rate**: `--lr 1e-4`
- **Modify reward shaping** in `ppo_agent/config.py`
- **Increase network size** in `config.py` (`hidden_dim = 512`)

### Agent Doesn't Learn

- Check that rewards are being collected (monitor logs)
- Ensure state normalization is working
- Try reducing clip epsilon: `config.clip_epsilon = 0.1`
- Increase entropy bonus: `config.entropy_coef = 0.05`

---

## 📚 Code Architecture

### PPO Agent (`ppo_agent/agent.py`)

Implements:
- `select_action()` - Sample actions from policy
- `store_transition()` - Add experience to buffer
- `update()` - Perform PPO update with clipped objective
- `save()/load()` - Model persistence

### Actor-Critic Model (`ppo_agent/model.py`)

- Shared feature extractor
- Separate policy and value heads
- Orthogonal weight initialization
- Categorical action distribution

### Rollout Buffer (`ppo_agent/memory.py`)

- Stores trajectories
- Computes GAE advantages
- Provides mini-batch sampling
- Advantage normalization

### Environment (`ppo_agent/env.py`)

- Gym-like interface
- State extraction and normalization
- Reward shaping
- Episode management

---

## 🎓 Learning Resources

### PPO Algorithm
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Original Paper
- [Spinning Up in Deep RL - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### Implementation Details
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

---

## 🏆 Expected Results

After training for **1 million timesteps**:
- Average survival: 400-600 steps
- Average kills: 8-15 zombies per episode
- Average reward: 100-180

After **2 million timesteps** (recommended):
- Average survival: 800-1200 steps
- Average kills: 20-35 zombies per episode
- Average reward: 250-400

---

## 🤝 Contributing

This is a complete, self-contained PPO implementation. Feel free to:
- Experiment with hyperparameters
- Add new reward shaping strategies
- Implement curriculum learning
- Add multi-agent support
- Integrate other RL algorithms

---

## 📝 License

This project is provided for educational purposes.

---

## 🙏 Acknowledgments

- PPO algorithm by OpenAI (Schulman et al., 2017)
- Pygame for the game engine
- PyTorch for deep learning

---

**Happy Training! 🚀**

For questions or issues, please review the troubleshooting section or check the code documentation.
