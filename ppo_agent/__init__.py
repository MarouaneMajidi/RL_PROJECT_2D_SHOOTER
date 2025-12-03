"""
PPO Agent Package for Top-Down Zombie Shooter

This package implements a complete Proximal Policy Optimization (PPO) agent
for learning to play a 2D zombie shooter game.
"""

from .agent import PPOAgent
from .model import ActorCritic
from .memory import RolloutBuffer
from .env import ZombieShooterEnv
from .config import PPOConfig

__all__ = [
    'PPOAgent',
    'ActorCritic',
    'RolloutBuffer',
    'ZombieShooterEnv',
    'PPOConfig'
]
