"""Utility helpers for trajectory normalization."""
from __future__ import annotations

import torch

from ml.data_generator import TRAJ_DIM

CONTACT_DIM = TRAJ_DIM - 1


def normalize_trajectory(
    traj: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    """Normalize trajectory except for the contact dimension."""
    if mean is None or std is None:
        return traj
    traj_norm = traj.clone()
    traj_norm[..., :CONTACT_DIM] = (traj_norm[..., :CONTACT_DIM] - mean[:CONTACT_DIM]) / std[:CONTACT_DIM]
    # Leave contact channel unchanged
    return traj_norm


def denormalize_trajectory(
    traj: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    """Denormalize trajectory except for the contact dimension."""
    if mean is None or std is None:
        return traj
    traj_out = traj.clone()
    traj_out[..., :CONTACT_DIM] = traj_out[..., :CONTACT_DIM] * std[:CONTACT_DIM] + mean[:CONTACT_DIM]
    return traj_out
