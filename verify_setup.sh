#!/bin/bash
# Verification script to check if PPO project is properly set up

echo "🔍 PPO Zombie Shooter - Setup Verification"
echo "=========================================="
echo ""

# Check Python
echo "✓ Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  Found: $PYTHON_VERSION"
else
    echo "  ✗ Python3 not found!"
    exit 1
fi
echo ""

# Check project structure
echo "✓ Checking project structure..."
REQUIRED_DIRS=("ppo_agent" "assets")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ exists"
    else
        echo "  ✗ $dir/ missing!"
        exit 1
    fi
done
echo ""

# Check PPO agent files
echo "✓ Checking PPO agent files..."
REQUIRED_FILES=(
    "ppo_agent/__init__.py"
    "ppo_agent/config.py"
    "ppo_agent/model.py"
    "ppo_agent/memory.py"
    "ppo_agent/env.py"
    "ppo_agent/agent.py"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file missing!"
        exit 1
    fi
done
echo ""

# Check main scripts
echo "✓ Checking main scripts..."
SCRIPTS=("main.py" "train_ppo.py" "test_agent.py")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script missing!"
        exit 1
    fi
done
echo ""

# Check documentation
echo "✓ Checking documentation..."
DOCS=("README.md" "TRAINING_GUIDE.md" "INSTRUCTIONS.md")
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✓ $doc"
    else
        echo "  ✗ $doc missing!"
    fi
done
echo ""

# Check Python syntax
echo "✓ Checking Python syntax..."
if python3 -m py_compile ppo_agent/*.py train_ppo.py test_agent.py main.py 2>/dev/null; then
    echo "  ✓ All Python files compile successfully"
else
    echo "  ✗ Syntax errors found!"
    exit 1
fi
echo ""

# Check dependencies
echo "✓ Checking dependencies..."
DEPS_MISSING=0

if python3 -c "import pygame" 2>/dev/null; then
    PYGAME_VERSION=$(python3 -c "import pygame; print(pygame.__version__)")
    echo "  ✓ pygame ($PYGAME_VERSION)"
else
    echo "  ✗ pygame not installed"
    DEPS_MISSING=1
fi

if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "  ✓ torch ($TORCH_VERSION)"
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        CUDA_STATUS=$(python3 -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')")
        echo "    CUDA available: $CUDA_STATUS"
    fi
else
    echo "  ✗ torch not installed"
    DEPS_MISSING=1
fi

if python3 -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    echo "  ✓ numpy ($NUMPY_VERSION)"
else
    echo "  ✗ numpy not installed"
    DEPS_MISSING=1
fi

if [ $DEPS_MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  Some dependencies are missing. Install with:"
    echo "   pip install pygame torch numpy"
    echo ""
fi
echo ""

# Final summary
echo "=========================================="
if [ $DEPS_MISSING -eq 0 ]; then
    echo "✅ Setup verification complete!"
    echo ""
    echo "🚀 You're ready to start training!"
    echo ""
    echo "Next steps:"
    echo "  1. Play manually:  python3 main.py --mode manual"
    echo "  2. Train agent:    python3 train_ppo.py"
    echo "  3. Test agent:     python3 test_agent.py --model checkpoints/best_model.pth"
    echo ""
    echo "See INSTRUCTIONS.md for detailed usage guide."
else
    echo "⚠️  Setup incomplete - install missing dependencies"
    echo ""
    echo "Run: pip install pygame torch numpy"
fi
echo "=========================================="
