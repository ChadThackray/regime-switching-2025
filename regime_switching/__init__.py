"""Regime-switching reinforcement learning module.

Implementation of continuous-time RL for optimal regime switching problems.
Based on the algorithm from arXiv:2512.04697v1.

Supports:
- Arbitrary number of regimes
- Arbitrary state dimension
- Custom dynamics and reward functions
"""

from .config import Config
from .environment import RegimeSwitchingEnv, RegimeSwitchingEnv2Regime
from .network import ValueNetwork
from .policy import compute_policy, compute_policy_2regime
from .simulation import simulate_trajectory
from .training import compute_martingale_loss, train

__all__ = [
    "Config",
    "RegimeSwitchingEnv",
    "RegimeSwitchingEnv2Regime",
    "ValueNetwork",
    "compute_policy",
    "compute_policy_2regime",
    "simulate_trajectory",
    "compute_martingale_loss",
    "train",
]
