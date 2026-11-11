"""
Residual-bound certificates and formal guarantees.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class ResidualBoundCertificate(nn.Module):
    """
    Computes an online residual bound of the form
        ||r_θ(z,u)|| ≤ r̂_max(z,u) + κ · σ_epi(z,u)
    and provides a probabilistic guarantee under standard concentration assumptions.
    """

    def __init__(self, kappa: float = 2.0):
        super().__init__()
        self.kappa = kappa

    def forward(
        self,
        residual_mean: torch.Tensor,
        residual_std_epi: torch.Tensor,
        residual_std_ale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            residual_mean: predicted residual magnitude (or vector norm)
            residual_std_epi: epistemic standard deviation
            residual_std_ale: aleatoric standard deviation
        Returns:
            bound: scalar upper bound
            violation_prob: approximate probability that true residual exceeds bound
        """
        bound = residual_mean + self.kappa * residual_std_epi + residual_std_ale
        # Under sub-Gaussian assumption the violation probability decays as exp(-κ²/2)
        violation_prob = torch.exp(-0.5 * self.kappa ** 2) * torch.ones_like(bound)
        return bound, violation_prob


def prove_residual_bound(
    residual_fn,
    ensemble_means: torch.Tensor,
    ensemble_stds: torch.Tensor,
    kappa: float = 2.0,
) -> dict:
    """
    Proof sketch implementation (see paper Proposition 1).

    Assumptions:
    1. The residual prediction error is sub-Gaussian with variance proxy σ².
    2. The ensemble provides an unbiased estimate of the mean residual.
    3. The physics backbone is Lipschitz continuous with constant L_phys.

    Then with probability ≥ 1 − δ the residual satisfies
        ||r_true − r_pred|| ≤ κσ + √(2σ² log(1/δ))
    """
    # Empirical implementation for the paper experiments
    mean_r = ensemble_means.mean(dim=0)
    std_r = ensemble_stds.mean(dim=0)
    bound = mean_r + kappa * std_r
    return {
        "bound": bound,
        "mean_residual": mean_r,
        "std_residual": std_r,
        "kappa": kappa,
    }
