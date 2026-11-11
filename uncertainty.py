"""
Epistemic (ensemble) + aleatoric uncertainty for residual bounds.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple
from .dynamics import ResidualNetwork


class EnsembleResidual(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, n_ensemble: int = 5):
        super().__init__()
        self.members = nn.ModuleList(
            [ResidualNetwork(latent_dim, action_dim) for _ in range(n_ensemble)]
        )
        self.n_ensemble = n_ensemble

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean residual
            epistemic std (across ensemble)
        """
        preds = torch.stack([m(z, u) for m in self.members], dim=0)  # [E, B, D]
        mean = preds.mean(0)
        std = preds.std(0)
        return mean, std


class AleatoricHead(nn.Module):
    """Predicts state-dependent aleatoric variance."""
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, u], dim=-1))
