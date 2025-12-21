# 🎮 PPO Zombie Shooter - Reinforcement Learning Project

A complete **Proximal Policy Optimization (PPO)** implementation from scratch that learns to play a 2D top-down zombie shooter game.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6+-green.svg)

## ✨ Features

- ✅ **PPO from scratch** - No Stable Baselines, pure PyTorch implementation
- ✅ **Actor-Critic architecture** with separate policy and value heads
- ✅ **GAE (Generalized Advantage Estimation)** for advantage computation
- ✅ **Clipped surrogate objective** for stable policy updates
- ✅ **Dual modes** - Play manually or watch the AI agent
- ✅ **Production-grade code** - Modular, documented, and extensible

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install pygame torch numpy

# Or using uv
uv sync
```

### Play the Game (Manual Mode)

```bash
python main.py --mode manual
```

### Train the PPO Agent

```bash
python train_ppo.py
```

### Watch the Trained Agent

```bash
python main.py --mode agent --model checkpoints/best_model.pth
```

### Test Agent Performance

```bash
python test_agent.py --model checkpoints/best_model.pth --episodes 10
```

## 📁 Project Structure

```
ppo_agent/
├── agent.py      # PPO algorithm implementation
├── model.py      # Actor-Critic neural network
├── memory.py     # Rollout buffer with GAE
├── env.py        # Gym-style environment wrapper
└── config.py     # Hyperparameters

train_ppo.py      # Training script
test_agent.py     # Testing script
main.py           # Game with manual/agent modes
```

## 🎯 PPO Algorithm Details

### State Space (98-dim)
- Player: position, velocity, health, cooldown, angle, zombie count
- Zombies: position, health, type, distance, angle (max 15 tracked)

### Action Space (6 discrete actions)
- Move Up/Down/Left/Right
- Shoot
- Idle

### Reward Function
- +10.0: Kill zombie
- +0.1: Survival per step
- -5.0: Damage taken (per 10 HP)
- -20.0: Death
- -0.01: Idle penalty

### Hyperparameters
- Learning Rate: 3e-4
- Discount (γ): 0.99
- GAE Lambda (λ): 0.95
- Clip Epsilon (ε): 0.2
- Batch Size: 64
- Rollout Steps: 2048

## 📊 Training

```bash
# Basic training (1M timesteps)
python train_ppo.py

# Extended training with custom settings
python train_ppo.py --timesteps 2000000 --lr 0.0003 --batch-size 64

# Train on CPU
python train_ppo.py --device cpu

# Resume from checkpoint
python train_ppo.py --checkpoint checkpoints/checkpoint_500000.pth
```

### Expected Results

**After 1M timesteps:**
- Survival: 400-600 steps
- Kills: 8-15 zombies
- Reward: 100-180

**After 2M timesteps:**
- Survival: 800-1200 steps
- Kills: 20-35 zombies
- Reward: 250-400

## 📚 Documentation

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive documentation including:
- Detailed installation instructions
- Training tips and hyperparameter tuning
- Algorithm explanation
- Troubleshooting guide
- Expected results and benchmarks

## 🔧 Customization

Edit `ppo_agent/config.py` to customize:
- Network architecture
- Training hyperparameters
- Reward shaping
- Environment settings

## 📝 Implementation Highlights

### PPO Agent (`ppo_agent/agent.py`)
- Clipped surrogate objective
- Mini-batch gradient updates
- Gradient clipping for stability
- Model checkpointing

### Actor-Critic Model (`ppo_agent/model.py`)
- Shared feature extractor (256-dim hidden layers)
- Separate policy and value heads
- Orthogonal weight initialization
- Categorical action distribution

### Rollout Buffer (`ppo_agent/memory.py`)
- Efficient trajectory storage
- GAE advantage computation
- Normalized advantages
- Mini-batch sampling

### Environment (`ppo_agent/env.py`)
- Gym-like interface
- State normalization
- Reward shaping
- Episode management

## 🏆 Performance Tips

1. **Train longer** - 2M timesteps recommended for best results
2. **Use GPU** - 5-10x faster training
3. **Monitor logs** - Check `logs/` for training progress
4. **Tune rewards** - Adjust reward coefficients in `config.py`
5. **Try different architectures** - Increase `hidden_dim` for more capacity

## 🐛 Troubleshooting

**CUDA Out of Memory?**
```bash
python train_ppo.py --batch-size 32 --device cpu
```

**Agent not learning?**
- Increase entropy bonus: `config.entropy_coef = 0.05`
- Reduce clip epsilon: `config.clip_epsilon = 0.1`
- Train longer: `--timesteps 2000000`

## 📖 Learning Resources

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original algorithm
- [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - OpenAI guide
- [37 Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Best practices

## 🙏 Acknowledgments

- PPO algorithm by Schulman et al. (2017)
- Pygame for game engine
- PyTorch for deep learning

## 📄 License

Educational project - feel free to use and modify!

---

**Built with ❤️ using PyTorch and Pygame**

For detailed instructions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
