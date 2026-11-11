"""
Training losses for PhysTwin: multi-step prediction + energy/momentum hinge.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, horizon: int = 10) -> torch.Tensor:
    """Multi-step MSE on proprioceptive state."""
    return ((pred - target) ** 2).mean()


def energy_momentum_hinge(
    energy_res: torch.Tensor,
    mom_res: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Hinge loss that penalizes violations of the residual bounds on
    energy and linear momentum.
    """
    e_hinge = torch.relu(energy_res - eps).mean()
    m_hinge = torch.relu(mom_res - eps).mean()
    return e_hinge + m_hinge


def total_loss(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    energy_res: torch.Tensor,
    mom_res: torch.Tensor,
    lambda_phys: float = 1.0,
    eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Combined objective used in the paper.
    """
    pred_l = prediction_loss(pred_traj, target_traj)
    phys_l = energy_momentum_hinge(energy_res, mom_res, eps)
    total = pred_l + lambda_phys * phys_l
    return {
        "total": total,
        "prediction": pred_l,
        "physics": phys_l,
    }
