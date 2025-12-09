"""Configuration for regime-switching RL problems."""

from dataclasses import dataclass


@dataclass
class Config:
    """Problem and training configuration for regime-switching RL.

    Problem-specific parameters (mu, sigma, g01, g10, temperature) have no defaults
    and must be provided. Training parameters have sensible defaults.
    """

    # Problem parameters (no defaults - must be specified)
    mu: tuple[float, float]  # Drift for each regime
    sigma: float  # Volatility
    g01: float  # Switching cost 0 -> 1
    g10: float  # Switching cost 1 -> 0
    temperature: float  # Lambda (exploration parameter)

    # Time discretization
    T: float = 1.0  # Time horizon
    K: int = 100  # Number of time steps

    # Training parameters
    batch_size: int = 64  # Number of trajectories per episode
    num_episodes: int = 1000  # Total training episodes
    learning_rate: float = 1e-3
    hidden_dim: int = 128

    # Initial state distribution
    x0_mean: float = 0.0
    x0_std: float = 2.0

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.K

    def switching_cost(self, i: int, j: int) -> float:
        """Get switching cost from regime i to regime j."""
        if i == j:
            return 0.0
        elif i == 0 and j == 1:
            return self.g01
        else:  # i == 1 and j == 0
            return self.g10
