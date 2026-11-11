"""
PhysTwin hybrid dynamics: physics backbone + constrained residual.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PhysicsBackbone(nn.Module):
    """Differentiable reduced-order rigid-body + simplified contact model.
    
    In the released code this is a placeholder that can be replaced by
    a MuJoCo-based analytical Jacobian or a learned reduced-order model
    that still exposes energy and momentum residuals.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Simple linear residual-free dynamics for demonstration.
        # Replace with real rigid-body equations for hardware.
        self.A = nn.Parameter(torch.eye(state_dim) * 0.95, requires_grad=False)
        self.B = nn.Parameter(torch.zeros(state_dim, action_dim), requires_grad=False)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return dx/dt from the physics model only."""
        return x @ self.A.T + u @ self.B.T

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Total mechanical energy (quadratic approximation)."""
        q, v = x.chunk(2, dim=-1)
        return 0.5 * (q**2).sum(-1) + 0.5 * (v**2).sum(-1)

    def momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Linear momentum of CoM (simplified)."""
        _, v = x.chunk(2, dim=-1)
        return v.sum(-1, keepdim=True)


class ResidualEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualNetwork(nn.Module):
    """Latent residual dynamics r_θ(z, u)."""
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),  # residual in latent space
        )
        # Map residual back to state space
        self.to_state = nn.Linear(latent_dim, latent_dim)  # placeholder; real code expands

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, u], dim=-1)
        return self.net(inp)


class PhysTwinDynamics(nn.Module):
    """Full hybrid model: f_phys + residual."""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 32):
        super().__init__()
        self.physics = PhysicsBackbone(state_dim, action_dim)
        self.encoder = ResidualEncoder(state_dim, latent_dim)
        self.residual = ResidualNetwork(latent_dim, action_dim)
        self.latent_dim = latent_dim
        self.state_dim = state_dim

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            dx: composite derivative
            z: latent encoding
        """
        z = self.encoder(x)
        r_latent = self.residual(z, u)
        # Project residual back (simplified; real implementation uses a decoder)
        r_state = torch.zeros_like(x)
        r_state[..., : self.latent_dim] = r_latent
        dx_phys = self.physics(x, u)
        return dx_phys + r_state, z

    def energy_residual(self, x: torch.Tensor, x_next: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        E_t = self.physics.energy(x)
        E_tp = self.physics.energy(x_next)
        work = dt * (u * x[..., self.state_dim // 2 :]).sum(-1)  # simplistic power
        return torch.abs(E_tp - E_t - work)

    def momentum_residual(self, x: torch.Tensor, x_next: torch.Tensor, dt: float) -> torch.Tensor:
        p_t = self.physics.momentum(x)
        p_tp = self.physics.momentum(x_next)
        return torch.norm(p_tp - p_t, dim=-1)
