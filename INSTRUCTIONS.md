# 🚀 QUICK START INSTRUCTIONS

## ✅ Project Successfully Built!

Your complete PPO reinforcement learning system is ready to use.

---

## 📦 What Was Created

### Core PPO Package (`ppo_agent/`)

1. **`config.py`** - All hyperparameters and settings
   - Learning rate, discount factor, GAE lambda
   - Network architecture settings
   - Reward shaping coefficients

2. **`model.py`** - Actor-Critic Neural Network
   - Shared feature extractor (256-dim)
   - Separate policy head (actor)
   - Separate value head (critic)
   - Orthogonal weight initialization

3. **`memory.py`** - Rollout Buffer
   - Stores trajectories (states, actions, rewards)
   - Computes GAE advantages
   - Provides mini-batch sampling

4. **`env.py`** - Gym-style Environment
   - Wraps Pygame game
   - State extraction (98-dim vector)
   - Action mapping (6 discrete actions)
   - Reward shaping

5. **`agent.py`** - PPO Algorithm
   - Clipped surrogate objective
   - Policy and value losses
   - Entropy bonus
   - Mini-batch updates
   - Gradient clipping

### Training & Testing Scripts

6. **`train_ppo.py`** - Training Loop
   - Collects rollouts
   - Performs PPO updates
   - Logs statistics
   - Saves checkpoints

7. **`test_agent.py`** - Testing Script
   - Loads trained models
   - Evaluates performance
   - Reports statistics

8. **`main.py`** - Modified Game
   - **Manual mode**: Human plays with keyboard/mouse
   - **Agent mode**: Trained PPO plays automatically

---

## 🎯 Step-by-Step Usage

### Step 1: Install Dependencies

```bash
# Option A: Using pip
pip install pygame torch numpy

# Option B: Using uv (faster)
uv sync

# For GPU support (recommended for faster training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Test the Game Manually

```bash
python3 main.py --mode manual
```

**Controls:**
- WASD or Arrow Keys: Move
- Mouse: Aim
- Left Click: Shoot
- 1: Pistol, 2: Machine Gun, 3: Knife
- R: Restart (when game over)
- Q: Quit

### Step 3: Train the PPO Agent

```bash
# Basic training (1 million timesteps, ~2-3 hours on GPU)
python3 train_ppo.py

# With custom settings
python3 train_ppo.py --timesteps 2000000 --lr 0.0003 --batch-size 64

# On CPU (slower)
python3 train_ppo.py --device cpu

# With visualization (much slower, for debugging)
python3 train_ppo.py --render
```

**What to expect:**
- Training creates `checkpoints/` and `logs/` directories
- Progress printed every update (~2048 timesteps)
- Best model saved to `checkpoints/best_model.pth`
- Training time: 2-4 hours for 1M timesteps on GPU

### Step 4: Watch the Trained Agent

```bash
# After training, watch the agent play
python3 main.py --mode agent --model checkpoints/best_model.pth
```

**Controls while watching:**
- ESC: Quit
- R: Restart episode

### Step 5: Evaluate Performance

```bash
# Test over 10 episodes
python3 test_agent.py --model checkpoints/best_model.pth --episodes 10

# Test without rendering (faster, for statistics)
python3 test_agent.py --model checkpoints/best_model.pth --episodes 100 --no-render
```

---

## 📊 Understanding Training Output

### During Training

```
Using device: cuda

Creating environment...
Creating PPO agent...

Starting training...
Total timesteps: 1,000,000

======================================================================
Timestep: 2,048 | Episode: 1
Episode Reward: 45.20 | Length: 512
Zombies Killed: 3

--- Update 1 ---
Timesteps: 2,048 / 1,000,000
Policy Loss: 0.0234      # Lower is better (policy improvement)
Value Loss: 0.4521       # Lower is better (value prediction)
Entropy: -1.7821         # Higher (less negative) = more exploration
Clip Fraction: 0.1234    # ~0.1-0.3 is good (policy change magnitude)
```

**Key Metrics:**
- **Episode Reward**: Total reward in episode (should increase over time)
- **Zombies Killed**: Combat effectiveness (should increase)
- **Policy Loss**: How much the policy is improving
- **Clip Fraction**: Fraction of updates that were clipped (0.1-0.3 ideal)

### After Training

Check logs in `logs/training_YYYYMMDD_HHMMSS.log` for detailed statistics.

---

## 🎮 Action Space Explanation

The agent can take 6 discrete actions each frame:

| Action | ID | Description |
|--------|----|----|
| Move Up | 0 | Move player upward |
| Move Down | 1 | Move player downward |
| Move Left | 2 | Move player left |
| Move Right | 3 | Move player right |
| Shoot | 4 | Fire weapon at nearest zombie |
| Idle | 5 | Do nothing |

The agent automatically aims at the nearest zombie when shooting.

---

## 📈 Expected Learning Curve

### Early Training (0-100k steps)
- Random exploration
- Low survival time (<100 steps)
- Few kills (0-2)
- Negative rewards

### Mid Training (100k-500k steps)
- Learning basic movement
- Survival: 200-400 steps
- Kills: 3-8
- Reward: 20-80

### Late Training (500k-1M steps)
- Effective combat
- Survival: 400-600 steps
- Kills: 8-15
- Reward: 100-180

### Extended Training (1M-2M steps)
- Mastery level
- Survival: 800-1200 steps
- Kills: 20-35
- Reward: 250-400

---

## ⚙️ Hyperparameter Tuning

Edit `ppo_agent/config.py` to customize:

```python
# For more exploration
config.entropy_coef = 0.05  # Default: 0.01

# For more stable updates
config.clip_epsilon = 0.1   # Default: 0.2

# For larger network
config.hidden_dim = 512     # Default: 256

# For different reward balance
config.reward_zombie_kill = 20.0        # Default: 10.0
config.reward_survival_per_step = 0.2   # Default: 0.1
```

---

## 🐛 Common Issues & Solutions

### Issue: "CUDA out of memory"

**Solution:**
```bash
python3 train_ppo.py --batch-size 32 --device cpu
```

### Issue: Agent doesn't learn (reward stays low)

**Solutions:**
1. Train longer: `--timesteps 2000000`
2. Increase entropy: Edit `config.py`, set `entropy_coef = 0.05`
3. Check reward shaping: Ensure zombies are spawning and rewards are collected
4. Reduce clip epsilon: Set `clip_epsilon = 0.1`

### Issue: Training is too slow

**Solutions:**
1. Use GPU: Ensure PyTorch with CUDA is installed
2. Don't render: Remove `--render` flag
3. Increase batch size: `--batch-size 128` (if GPU memory allows)

### Issue: "Module 'ppo_agent' not found"

**Solution:**
```bash
# Make sure you're in the project directory
cd /home/sohaib/Desktop/RL_PROJECT_2D_SHOOTER

# Verify ppo_agent/ exists
ls ppo_agent/
```

---

## 📁 Project Files Overview

```
RL_PROJECT_2D_SHOOTER/
│
├── ppo_agent/              # PPO implementation
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Hyperparameters
│   ├── model.py           # Neural network
│   ├── memory.py          # Replay buffer
│   ├── env.py             # Environment wrapper
│   └── agent.py           # PPO algorithm
│
├── main.py                # Game (manual/agent modes)
├── train_ppo.py           # Training script
├── test_agent.py          # Testing script
│
├── assets/                # Game graphics
│   ├── player pistol.gif
│   ├── zombie.gif
│   └── ...
│
├── checkpoints/           # Saved models (created during training)
│   ├── best_model.pth
│   ├── final_model.pth
│   └── checkpoint_*.pth
│
├── logs/                  # Training logs (created during training)
│   └── training_*.log
│
├── README.md              # Quick overview
├── TRAINING_GUIDE.md      # Comprehensive guide
└── INSTRUCTIONS.md        # This file
```

---

## 🏆 Training Recommendations

For best results:

1. **Start with default settings**
   ```bash
   python3 train_ppo.py --timesteps 1000000
   ```

2. **Monitor first 50k steps**
   - Rewards should become less negative
   - Zombies killed should increase from 0 to 2-3

3. **If learning is too slow:**
   - Increase entropy: `config.entropy_coef = 0.02`
   - Reduce learning rate: `--lr 0.0001`

4. **If learning is unstable:**
   - Reduce clip epsilon: `config.clip_epsilon = 0.1`
   - Reduce learning rate: `--lr 0.0001`

5. **For best performance:**
   - Train for 2M timesteps
   - Use GPU
   - Monitor logs and adjust rewards if needed

---

## 📚 Next Steps

1. **Train your first agent:**
   ```bash
   python3 train_ppo.py --timesteps 500000
   ```

2. **Test it:**
   ```bash
   python3 test_agent.py --model checkpoints/best_model.pth
   ```

3. **Watch it play:**
   ```bash
   python3 main.py --mode agent --model checkpoints/best_model.pth
   ```

4. **Experiment:**
   - Modify rewards in `config.py`
   - Try different network sizes
   - Adjust exploration (entropy coefficient)

5. **Read the guide:**
   - See `TRAINING_GUIDE.md` for detailed explanations
   - Learn about PPO algorithm internals
   - Understand reward shaping strategies

---

## 💡 Pro Tips

1. **Use GPU for training** - 5-10x faster
2. **Save checkpoints frequently** - Don't lose progress
3. **Monitor clip fraction** - Should be 0.1-0.3 for healthy learning
4. **Watch the agent periodically** - Visual inspection helps debug
5. **Keep training logs** - Useful for analysis and tuning
6. **Start with shorter training** - 500k steps to verify everything works
7. **Experiment with rewards** - Biggest impact on behavior

---

## 🎓 Understanding PPO

**PPO (Proximal Policy Optimization)** is a state-of-the-art RL algorithm that:
- Learns both a policy (what to do) and a value function (how good is a state)
- Uses "clipping" to prevent too-large policy updates
- Balances exploration and exploitation
- Is sample efficient and stable

**Key Components:**
1. **Actor** - Decides which action to take
2. **Critic** - Estimates how good the current state is
3. **GAE** - Computes advantages (how much better is this action than average)
4. **Clipping** - Prevents destructive policy updates

---

## 🔗 Helpful Resources

- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Spinning Up Guide**: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- **Implementation Details**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

---

## ✅ Checklist

Before training:
- [ ] Dependencies installed (`pygame`, `torch`, `numpy`)
- [ ] Can run manual mode: `python3 main.py --mode manual`
- [ ] Project structure looks correct
- [ ] GPU available (optional but recommended)

Ready to train:
- [ ] Run: `python3 train_ppo.py`
- [ ] Monitor output for learning progress
- [ ] Check `checkpoints/` for saved models
- [ ] Check `logs/` for training statistics

After training:
- [ ] Test agent: `python3 test_agent.py --model checkpoints/best_model.pth`
- [ ] Watch agent: `python3 main.py --mode agent --model checkpoints/best_model.pth`
- [ ] Analyze logs for insights
- [ ] Experiment with hyperparameters

---

## 🎉 You're Ready!

Everything is set up and ready to go. Start with:

```bash
python3 train_ppo.py
```

And let the agent learn to survive the zombie apocalypse! 🧟‍♂️🔫

For questions or issues, consult:
1. This file (INSTRUCTIONS.md)
2. Detailed guide (TRAINING_GUIDE.md)
3. Code comments in the source files

**Happy Training! 🚀**
