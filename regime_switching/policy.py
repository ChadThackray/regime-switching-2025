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
    Compute switching intensities from value function (eq 3.5 in paper).

    π*_ij(t,x) = exp((V_j(t,x) - g_ij - V_i(t,x)) / λ)

    For m regimes, returns the intensity of switching to each OTHER regime.

    Args:
        value_net: Value function network
        t: Time tensor (batch,)
        x: State tensor (batch,) or (batch, state_dim)
        i: Current regime indices (batch,)
        config: Problem configuration

    Returns:
        pi_row: Switching intensities to all other regimes (batch, num_regimes)
            Entry [b, j] is intensity of switching from regime i[b] to regime j.
            Diagonal entries (j == i[b]) are 0.
        entropy: Entropy term R(π, i) for the objective (batch,)
    """
    num_regimes = config.num_regimes
    batch_size = t.shape[0]
    device = t.device

    # Get values for all regimes: (batch, num_regimes)
    values = value_net(t, x)

    # Get switching cost matrix: (num_regimes, num_regimes)
    g = config.get_switching_cost_matrix(device)

    # Get value at current regime for each sample: (batch,)
    v_i = values.gather(1, i.unsqueeze(1)).squeeze(1)

    # Compute π_ij = exp((V_j - g_ij - V_i) / λ) for all j
    # v_i: (batch,) -> (batch, 1) for broadcasting
    # g[i, :]: need to select row i for each batch element
    # g shape: (num_regimes, num_regimes)
    # i shape: (batch,)

    # Get g[i[b], j] for all b, j: (batch, num_regimes)
    g_i = g[i]  # (batch, num_regimes)

    # Compute argument: (V_j - g_ij - V_i) / λ
    # values: (batch, num_regimes) - V_j for all j
    # g_i: (batch, num_regimes) - g[i,j] for all j
    # v_i: (batch,) -> (batch, 1)
    arg = (values - g_i - v_i.unsqueeze(1)) / config.temperature  # (batch, num_regimes)

    # Compute intensities
    pi_row = torch.exp(arg)  # (batch, num_regimes)

    # Zero out diagonal (no self-transitions)
    # Create mask where diagonal elements (j == i) are False
    j_indices = (
        torch.arange(num_regimes, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    i_expanded = i.unsqueeze(1).expand(-1, num_regimes)
    diagonal_mask = j_indices == i_expanded  # (batch, num_regimes)
    pi_row = pi_row.masked_fill(diagonal_mask, 0.0)

    # Clamp to prevent numerical issues
    pi_row = pi_row.clamp(min=0.0, max=20.0)

    # Compute entropy: R(π, i) = Σ_{j≠i} (π_ij - π_ij * log(π_ij))
    # Use safe log to avoid log(0)
    safe_pi = pi_row.clamp(min=1e-10)
    entropy_terms = pi_row - pi_row * torch.log(safe_pi)  # (batch, num_regimes)
    # Zero out diagonal contributions (already 0 from pi_row, but be safe)
    entropy_terms = entropy_terms.masked_fill(diagonal_mask, 0.0)
    entropy = entropy_terms.sum(dim=1)  # (batch,)

    return pi_row, entropy


def compute_policy_2regime(
    value_net: ValueNetwork,
    t: torch.Tensor,
    x: torch.Tensor,
    i: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized policy computation for 2-regime case (backward compatible).

    Returns scalar switching intensity (to the OTHER regime) instead of full row.

    Args:
        value_net: Value function network
        t: Time tensor (batch,)
        x: State tensor (batch,)
        i: Current regime indices (batch,) - values in {0, 1}
        config: Problem configuration

    Returns:
        pi_switch: Switching intensity to the other regime (batch,)
        entropy: Entropy term R(π, i) for the objective (batch,)
    """
    if config.num_regimes != 2:
        raise ValueError("compute_policy_2regime only works for 2 regimes")

    values = value_net(t, x)  # (batch, 2)
    v_0 = values[:, 0]
    v_1 = values[:, 1]

    g01 = config.switching_cost(0, 1)
    g10 = config.switching_cost(1, 0)

    # Compute switching intensities
    # π_01 = exp((v_1 - g_01 - v_0) / λ)
    pi_01 = torch.exp((v_1 - g01 - v_0) / config.temperature)
    # π_10 = exp((v_0 - g_10 - v_1) / λ)
    pi_10 = torch.exp((v_0 - g10 - v_1) / config.temperature)

    # Select based on current regime
    pi_switch = torch.where(i == 0, pi_01, pi_10)

    # Clamp to prevent numerical blow-up
    pi_switch = pi_switch.clamp(min=1e-10, max=20.0)

    # Entropy: R(π, i) = π - π * log(π) for single transition
    entropy = pi_switch - pi_switch * torch.log(pi_switch)

    return pi_switch, entropy
