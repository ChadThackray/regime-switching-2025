"""Regime-switching reinforcement learning module.

Implementation of continuous-time RL for optimal regime switching problems.
Based on the algorithm from arXiv:2512.04697v1.
"""

from .config import Config
from .environment import RegimeSwitchingEnv
from .network import ValueNetwork
from .policy import compute_policy
from .simulation import simulate_trajectory
from .training import compute_martingale_loss, train

__all__ = [
    "Config",
    "RegimeSwitchingEnv",
    "ValueNetwork",
    "compute_policy",
    "simulate_trajectory",
    "compute_martingale_loss",
    "train",
]
