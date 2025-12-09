"""Configuration for regime-switching RL problems."""

from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    """Problem and training configuration for regime-switching RL.

    This is a generalized config supporting:
    - Arbitrary number of regimes (num_regimes)
    - Arbitrary state dimension (state_dim)
    - Full switching cost matrix

    Problem-specific dynamics and rewards are passed as functions to the training loop.
    """

    # Problem structure
    num_regimes: int = 2
    state_dim: int = 1

    # Switching costs: m x m matrix where g[i][j] is cost to switch from i to j
    # Diagonal entries (g[i][i]) should be 0
    switching_costs: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.5], [0.5, 0.0]]
    )

    # Temperature (exploration parameter)
    temperature: float = 0.2

    # Time discretization
    T: float = 1.0  # Time horizon
    K: int = 100  # Number of time steps

    # Training parameters
    batch_size: int = 64  # Number of trajectories per episode
    num_episodes: int = 1000  # Total training episodes
    learning_rate: float = 1e-3
    hidden_dim: int = 128

    # Initial state distribution (for 1D; higher dims handled by init_state_fn)
    x0_mean: float = 0.0
    x0_std: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration and convert switching costs to tensor-friendly format."""
        # Validate switching cost matrix dimensions
        if len(self.switching_costs) != self.num_regimes:
            raise ValueError(
                f"switching_costs has {len(self.switching_costs)} rows, "
                f"expected {self.num_regimes}"
            )
        for i, row in enumerate(self.switching_costs):
            if len(row) != self.num_regimes:
                raise ValueError(
                    f"switching_costs row {i} has {len(row)} columns, "
                    f"expected {self.num_regimes}"
                )

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.K

    def switching_cost(self, i: int, j: int) -> float:
        """Get switching cost from regime i to regime j."""
        return self.switching_costs[i][j]

    def get_switching_cost_matrix(self, device: torch.device) -> torch.Tensor:
        """Get switching cost matrix as a tensor."""
        return torch.tensor(self.switching_costs, device=device, dtype=torch.float32)
