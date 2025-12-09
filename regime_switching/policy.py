"""Policy computation from value function."""

import torch

from .config import Config
from .network import ValueNetwork


def compute_policy(
    value_net: ValueNetwork,
    t: torch.Tensor,
    x: torch.Tensor,
    i: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute switching intensity from value function (eq 3.5 in paper).

    pi*_ij(t,x) = exp((V_j(t,x) - g_ij - V_i(t,x)) / lambda)

    Args:
        value_net: Value function network
        t: Time tensor (batch,)
        x: State tensor (batch,)
        i: Current regime indices (batch,)
        config: Problem configuration

    Returns:
        pi_switch: Switching intensity to the other regime (batch,)
        entropy: Entropy term R(pi, i) for the objective (batch,)
    """
    values = value_net(t, x)  # (batch, 2)
    v_0 = values[:, 0]
    v_1 = values[:, 1]

    # Compute switching intensities
    # pi_01 = exp((v_1 - g_01 - v_0) / lambda)
    pi_01 = torch.exp((v_1 - config.g01 - v_0) / config.temperature)
    # pi_10 = exp((v_0 - g_10 - v_1) / lambda)
    pi_10 = torch.exp((v_0 - config.g10 - v_1) / config.temperature)

    # Select the appropriate switching intensity based on current regime
    pi_switch = torch.where(i == 0, pi_01, pi_10)

    # Clamp to prevent numerical blow-up in simulation
    pi_switch = pi_switch.clamp(min=1e-10, max=20.0)

    # Compute entropy term: R(pi, i) = sum_{j!=i} (pi_ij - pi_ij * log(pi_ij))
    entropy = pi_switch - pi_switch * torch.log(pi_switch)

    return pi_switch, entropy
