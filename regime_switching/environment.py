"""Environment for simulating regime-switching diffusion."""

from collections.abc import Callable

import numpy as np
import torch

from .config import Config


class RegimeSwitchingEnv:
    """
    Environment for simulating regime-switching diffusion.

    State dynamics: dX_t = drift(t, X, i) dt + diffusion(t, X, i) dW_t
    Regime transitions governed by CTMC with controlled intensity.

    Supports:
    - Arbitrary state dimension
    - Arbitrary number of regimes
    - Custom dynamics and reward functions
    """

    def __init__(
        self,
        config: Config,
        dynamics_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        running_reward_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        device: torch.device,
    ):
        """
        Initialize environment.

        Args:
            config: Problem configuration
            dynamics_fn: Function (t, x, i) -> (drift, diffusion)
                - t: (batch,)
                - x: (batch,) or (batch, state_dim)
                - i: (batch,) regime indices
                - drift: (batch,) or (batch, state_dim)
                - diffusion: (batch,) or (batch, state_dim)
            running_reward_fn: Function (t, x, i) -> reward
                - t: (batch,)
                - x: (batch,) or (batch, state_dim)
                - i: (batch,) regime indices
                - reward: (batch,)
            device: Torch device
        """
        self.config = config
        self.dynamics_fn = dynamics_fn
        self.running_reward_fn = running_reward_fn
        self.device = device

    def step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        i: torch.Tensor,
        pi_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one time step.

        Args:
            t: Current time (batch,)
            x: Current state (batch,) or (batch, state_dim)
            i: Current regime (batch,)
            pi_row: Switching intensities (batch, num_regimes)
                Entry [b, j] is intensity of switching from i[b] to j.
                Diagonal entries should be 0 (no self-transitions).

        Returns:
            next_x: Next state (batch,) or (batch, state_dim)
            next_i: Next regime (batch,)
            reward: Running reward for this step (batch,)
            switch_cost: Switching cost incurred (batch,)
        """
        batch_size = t.shape[0]
        dt = self.config.dt

        # Get dynamics
        drift, diffusion = self.dynamics_fn(t, x, i)

        # Simulate SDE: X_{t+dt} = X_t + drift * dt + diffusion * dW
        if x.dim() == 1:
            dW = torch.randn(batch_size, device=self.device) * np.sqrt(dt)
            next_x = x + drift * dt + diffusion * dW
        else:
            state_dim = x.shape[1]
            dW = torch.randn(batch_size, state_dim, device=self.device) * np.sqrt(dt)
            next_x = x + drift * dt + diffusion * dW

        # Simulate regime switching via CTMC
        # Total switching rate: λ_i = Σ_{j≠i} π_ij
        total_rate = pi_row.sum(dim=1)  # (batch,)

        # Probability of any switch in interval dt: 1 - exp(-λ_i * dt)
        switch_prob = 1.0 - torch.exp(-total_rate * dt)
        switch_random = torch.rand(batch_size, device=self.device)
        will_switch = switch_random < switch_prob  # (batch,)

        # If switching, choose target regime proportional to intensities
        # P(switch to j | switch) = π_ij / λ_i
        # Sample from categorical distribution over regimes
        # (excluding current regime, but pi_row[current] = 0 anyway)

        # Normalize to get probabilities
        # Use total_rate.clamp to avoid div by zero
        probs = pi_row / total_rate.unsqueeze(1).clamp(
            min=1e-10
        )  # (batch, num_regimes)

        # Sample target regime for each batch element
        target_regime = torch.multinomial(
            probs.clamp(min=1e-10), num_samples=1
        ).squeeze(1)  # (batch,)

        # Apply switch only where will_switch is True
        next_i = torch.where(will_switch, target_regime, i)

        # Running reward (based on state before transition)
        reward = self.running_reward_fn(t, x, i)

        # Compute switching cost
        # Get g[i[b], next_i[b]] for each batch element
        g = self.config.get_switching_cost_matrix(
            self.device
        )  # (num_regimes, num_regimes)
        # Only charge cost if actually switched
        switch_cost = torch.where(
            will_switch,
            g[i, next_i],
            torch.zeros(batch_size, device=self.device),
        )

        return next_x, next_i, reward, switch_cost


class RegimeSwitchingEnv2Regime:
    """
    Optimized environment for 2-regime case (backward compatible).

    Uses scalar switching intensity instead of full intensity row.
    """

    def __init__(
        self,
        config: Config,
        dynamics_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        running_reward_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        device: torch.device,
    ):
        """
        Initialize environment.

        Args:
            config: Problem configuration
            dynamics_fn: Function (t, x, i) -> (drift, diffusion)
            running_reward_fn: Function (t, x, i) -> reward
            device: Torch device
        """
        if config.num_regimes != 2:
            raise ValueError("RegimeSwitchingEnv2Regime only works for 2 regimes")
        self.config = config
        self.dynamics_fn = dynamics_fn
        self.running_reward_fn = running_reward_fn
        self.device = device

    def step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        i: torch.Tensor,
        pi_switch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one time step.

        Args:
            t: Current time (batch,)
            x: Current state (batch,)
            i: Current regime (batch,) - values in {0, 1}
            pi_switch: Switching intensity to the OTHER regime (batch,)

        Returns:
            next_x: Next state (batch,)
            next_i: Next regime (batch,)
            reward: Running reward for this step (batch,)
            switch_cost: Switching cost incurred (batch,)
        """
        batch_size = x.shape[0]
        dt = self.config.dt

        # Get dynamics
        drift, diffusion = self.dynamics_fn(t, x, i)

        # Simulate SDE
        dW = torch.randn(batch_size, device=self.device) * np.sqrt(dt)
        next_x = x + drift * dt + diffusion * dW

        # Simulate regime switching
        switch_prob = 1.0 - torch.exp(-pi_switch * dt)
        switch_random = torch.rand(batch_size, device=self.device)
        switched = switch_random < switch_prob

        # If switched, flip regime (0 -> 1 or 1 -> 0)
        next_i = torch.where(switched, 1 - i, i)

        # Running reward
        reward = self.running_reward_fn(t, x, i)

        # Switching cost
        g01 = self.config.switching_cost(0, 1)
        g10 = self.config.switching_cost(1, 0)
        cost = torch.zeros(batch_size, device=self.device)
        switched_01 = switched & (i == 0)
        switched_10 = switched & (i == 1)
        cost = torch.where(switched_01, torch.tensor(g01, device=self.device), cost)
        cost = torch.where(switched_10, torch.tensor(g10, device=self.device), cost)

        return next_x, next_i, reward, cost
