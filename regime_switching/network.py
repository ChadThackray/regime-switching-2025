"""Neural network for value function approximation."""

from collections.abc import Callable

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Neural network to approximate the value function v(t, x, i).

    Uses a structural decomposition to enforce terminal condition:
    v(t, x, i) = h(x, i) + (T - t) * phi(t, x, i)

    This ensures v(T, x, i) = h(x, i) exactly.

    Supports:
    - Arbitrary state dimension (state_dim)
    - Arbitrary number of regimes (num_regimes)
    """

    def __init__(
        self,
        terminal_reward: Callable[[torch.Tensor], torch.Tensor],
        hidden_dim: int = 128,
        T: float = 1.0,
        state_dim: int = 1,
        num_regimes: int = 2,
    ):
        """
        Initialize value network.

        Args:
            terminal_reward: Function h(x) that computes terminal reward.
                Should return shape (batch,) or (batch, num_regimes).
                If (batch,), the same value is used for all regimes.
            hidden_dim: Hidden layer dimension
            T: Time horizon
            state_dim: Dimension of state space
            num_regimes: Number of regimes
        """
        super().__init__()
        self.T = T
        self.terminal_reward = terminal_reward
        self.state_dim = state_dim
        self.num_regimes = num_regimes

        # Network outputs the "correction" phi for all regimes
        # Input: (t, x_1, ..., x_n) -> state_dim + 1 features
        # Output: (phi_0, phi_1, ..., phi_{m-1}) -> num_regimes values
        input_dim = state_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_regimes),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute value function for all regimes.

        v(t, x, i) = h(x, i) + (T - t) * phi(t, x, i)

        Args:
            t: Time tensor of shape (batch,)
            x: State tensor of shape (batch,) for 1D or (batch, state_dim) for multi-dim

        Returns:
            Values tensor of shape (batch, num_regimes) - one column per regime
        """
        # Handle 1D state (backward compatibility)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (batch,) -> (batch, 1)

        # Concatenate time and state
        t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t  # (batch, 1)
        inputs = torch.cat([t_expanded, x], dim=-1)  # (batch, state_dim + 1)
        phi = self.net(inputs)  # (batch, num_regimes)

        # Terminal reward - handle both scalar and per-regime versions
        h = self.terminal_reward(x.squeeze(-1) if self.state_dim == 1 else x)
        if h.dim() == 1:
            # Scalar terminal reward - broadcast to all regimes
            h = h.unsqueeze(-1).expand_as(phi)  # (batch, num_regimes)

        # Time-to-maturity weighting ensures v(T, x, i) = h(x, i)
        time_factor = (self.T - t).unsqueeze(-1).expand_as(phi)  # (batch, num_regimes)

        return h + time_factor * phi

    def get_value(
        self, t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
    ) -> torch.Tensor:
        """Get value for specific regime indices.

        Args:
            t: Time tensor (batch,)
            x: State tensor (batch,) or (batch, state_dim)
            i: Regime indices (batch,)

        Returns:
            Values tensor (batch,)
        """
        all_values = self.forward(t, x)  # (batch, num_regimes)
        return all_values.gather(1, i.unsqueeze(1)).squeeze(1)
