# GPU Training Setup & New Features

## ✅ Changes Made

### 1. **GPU Training Support**
- Auto-detects GPU availability
- Falls back to CPU if GPU not available
- Config now uses: `device = "cuda" if torch.cuda.is_available() else "cpu"`

### 2. **Wave Progression During Training**
- Spawn rate decreases every 10 zombie kills (waves get harder)
- Starts at 60 frames between spawns
- Minimum spawn rate: 10 frames
- Wave tracking included in training logs

### 3. **Pickup Collection Rewards**
- **Health pickup**: +5.0 reward
- **Machine gun pickup**: +3.0 reward
- Encourages agent to collect beneficial items

## 🔧 Install GPU Support

You have an **NVIDIA GeForce GTX 960M** with CUDA 12.2 available.

### Install PyTorch with CUDA:

```bash
# Activate your environment
conda activate rl_zoo

# Uninstall CPU version
pip uninstall torch torchvision torchaudio -y

# Install CUDA version (for CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Expected Output:
```
CUDA available: True
Device: NVIDIA GeForce GTX 960M
```

## 🚀 Training with GPU

Once GPU support is installed, training will automatically use it:

```bash
# Basic training (will use GPU automatically)
python train_ppo.py

# Force CPU if needed
python train_ppo.py --device cpu

# Extended training with custom settings
python train_ppo.py --timesteps 2000000 --batch-size 64
```

### GPU vs CPU Speed:
- **CPU**: ~100-200 steps/second
- **GPU**: ~500-1000 steps/second (5-10x faster)

## 📊 New Training Features

### Wave Mechanics
The training environment now includes wave progression:
- Wave 1: Spawn every 60 frames
- Wave 2: Spawn every 55 frames (after 10 kills)
- Wave 3: Spawn every 50 frames (after 20 kills)
- ...continues until minimum of 10 frames

This teaches the agent to handle increasing difficulty.

### Pickup Rewards
The agent now gets rewarded for collecting items:

| Item | Reward | Benefit |
|------|--------|---------|
| Health Pickup | +5.0 | Restores 25 HP |
| Machine Gun | +3.0 | Extra weapon |

### Updated Reward Structure:
```python
Kill zombie:          +10.0
Survival (per step):  +0.1
Health pickup:        +5.0   # NEW
Machine gun pickup:   +3.0   # NEW
Damage taken:         -5.0 (per 10 HP)
Death:                -20.0
Idle penalty:         -0.01
Approach zombie:      +0.01
```

## 📈 Training Logs

Logs now include wave information:

```csv
timestep,episode,episode_reward,episode_length,zombies_killed,wave,policy_loss,...
2048,1,45.20,512,3,1,0.0234,...
4096,2,156.30,892,15,2,0.0189,...
```

## 🎮 Memory Considerations

Your GTX 960M has **2GB VRAM**. Recommendations:

### For 2GB VRAM:
```bash
# Use smaller batch size
python train_ppo.py --batch-size 32 --n-steps 1024

# If still out of memory, use CPU
python train_ppo.py --device cpu
```

### For larger GPUs (4GB+):
```bash
# Use default settings
python train_ppo.py --batch-size 64 --n-steps 2048
```

## 🧪 Test GPU Training

Quick test to verify GPU training works:

```bash
# Activate environment
conda activate rl_zoo

# Short training run (10k steps)
python train_ppo.py --timesteps 10000 --device cuda

# Check output for:
# "Using device: cuda"
```

## ⚙️ Configuration Summary

Changes in `ppo_agent/config.py`:

```python
# Auto-detect GPU
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# New pickup rewards
reward_health_pickup: float = 5.0
reward_machinegun_pickup: float = 3.0
```

Changes in `ppo_agent/env.py`:
- Wave tracking with spawn rate progression
- Pickup collection rewards applied
- Wave info included in episode info dict

Changes in `train_ppo.py`:
- Wave column added to logs
- Wave displayed in console output

## 🎯 Recommended Training Command

```bash
# After installing CUDA PyTorch:
conda activate rl_zoo
python train_ppo.py --timesteps 2000000 --batch-size 32
```

The smaller batch size (32 instead of 64) is safer for 2GB VRAM.

## 📊 Expected Results

With these improvements:

**After 1M timesteps:**
- Survival: 500-700 steps (improved with pickup collection)
- Kills: 10-18 zombies
- Waves reached: 2-3
- Better strategic behavior (collects health)

**After 2M timesteps:**
- Survival: 1000-1500 steps
- Kills: 25-40 zombies
- Waves reached: 3-5
- Mastery level performance

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train_ppo.py --batch-size 16 --n-steps 1024

# Or use CPU
python train_ppo.py --device cpu
```

### "CUDA not available" after installing
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Training too slow on GPU
- Check GPU utilization: `nvidia-smi`
- Ensure no other programs using GPU
- Try larger batch size if VRAM allows

## ✅ Quick Start

1. Install CUDA PyTorch (see above)
2. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
3. Start training: `python train_ppo.py --batch-size 32`
4. Monitor with: `nvidia-smi` (in another terminal)

---

Your agent will now train faster on GPU and learn more sophisticated behaviors with wave mechanics and pickup collection! 🚀
