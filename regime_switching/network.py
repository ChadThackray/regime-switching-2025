"""Neural network for value function approximation."""

from collections.abc import Callable

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Neural network to approximate the value function v(t, x, i).

    Uses a structural decomposition to enforce terminal condition:
    v(t, x, i) = h(x) + (T - t) * phi(t, x, i)

    This ensures v(T, x, i) = h(x) exactly.
    """

    def __init__(
        self,
        terminal_reward: Callable[[torch.Tensor], torch.Tensor],
        hidden_dim: int = 128,
        T: float = 1.0,
    ):
        """
        Initialize value network.

        Args:
            terminal_reward: Function h(x) that computes terminal reward
            hidden_dim: Hidden layer dimension
            T: Time horizon
        """
        super().__init__()
        self.T = T
        self.terminal_reward = terminal_reward

        # Network outputs the "correction" phi for both regimes
        # Input: (t, x) -> 2 features
        # Output: (phi_0, phi_1) -> 2 values
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute value function for both regimes.

        v(t, x, i) = h(x) + (T - t) * phi(t, x, i)

        Args:
            t: Time tensor of shape (batch,)
            x: State tensor of shape (batch,)

        Returns:
            Values tensor of shape (batch, 2) - one column per regime
        """
        inputs = torch.stack([t, x], dim=-1)
        phi = self.net(inputs)  # (batch, 2)

        # Terminal reward (same for both regimes)
        h = self.terminal_reward(x).unsqueeze(-1).expand_as(phi)  # (batch, 2)

        # Time-to-maturity weighting ensures v(T, x, i) = h(x)
        time_factor = (self.T - t).unsqueeze(-1).expand_as(phi)  # (batch, 2)

        return h + time_factor * phi

    def get_value(self, t: torch.Tensor, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Get value for specific regime indices."""
        all_values = self.forward(t, x)  # (batch, 2)
        return all_values.gather(1, i.unsqueeze(1)).squeeze(1)
