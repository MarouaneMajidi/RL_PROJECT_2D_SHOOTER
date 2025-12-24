"""
Rollout Buffer for CNN-based DDA Agent

Stores image-based trajectories and computes advantages using GAE.
States are images with shape (12, 96, 128) instead of 1D vectors.
"""

# Reuse the same CNNRolloutBuffer from player_cnn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_cnn.memory import CNNRolloutBuffer

# Alias for DDA use
DDACNNRolloutBuffer = CNNRolloutBuffer
