"""Environment for simulating regime-switching diffusion."""

from collections.abc import Callable

import numpy as np
import torch

from .config import Config


class RegimeSwitchingEnv:
    """
    Environment for simulating regime-switching diffusion.

    State dynamics: dX_t = mu_i dt + sigma dW_t
    Regime transitions governed by CTMC with controlled intensity.
    """

    def __init__(
        self,
        config: Config,
        running_reward: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ):
        """
        Initialize environment.

        Args:
            config: Problem configuration
            running_reward: Function f(x) that computes running reward
            device: Torch device
        """
        self.config = config
        self.running_reward = running_reward
        self.device = device
        self.mu = torch.tensor(config.mu, device=device)

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
            switched: Whether a switch occurred (batch,) - bool
        """
        batch_size = x.shape[0]
        dt = self.config.dt
        sigma = self.config.sigma

        # Get drift based on current regime
        drift = self.mu[i]  # (batch,)

        # Simulate SDE: X_{t+dt} = X_t + mu_i * dt + sigma * dW
        dW = torch.randn(batch_size, device=self.device) * np.sqrt(dt)
        next_x = x + drift * dt + sigma * dW

        # Simulate regime switching via CTMC
        # Probability of switching in interval dt: 1 - exp(-pi * dt) ~ pi * dt
        switch_prob = 1.0 - torch.exp(-pi_switch * dt)
        switch_random = torch.rand(batch_size, device=self.device)
        switched = switch_random < switch_prob

        # If switched, flip regime (0 -> 1 or 1 -> 0)
        next_i = torch.where(switched, 1 - i, i)

        # Running reward
        reward = self.running_reward(x)

        return next_x, next_i, reward, switched
