# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from eb_jepa/eb_jepa/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeStdLoss(nn.Module):
    """Encourages each feature to maintain at least a minimum standard deviation.
    Features with std below the margin incur a penalty of (std_margin - std).

    Args:
        std_margin (float, default=1.0): Minimum desired standard deviation per feature.
    """

    def __init__(self, std_margin: float = 1.0):
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(nn.Module):
    """Penalizes off-diagonal elements of the covariance matrix to encourage
    feature decorrelation."""

    def __init__(self):
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        cov_loss = self.off_diagonal(cov).pow(2).mean()
        return cov_loss


class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std), and covariance terms.

    Args:
        inv_coeff (float): Weight for invariance (MSE) loss.
        std_coeff (float): Weight for variance (hinge std) loss.
        cov_coeff (float): Weight for covariance (decorrelation) loss.
    """

    def __init__(self, inv_coeff=1.0, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """Compute VICReg loss.

        Args:
            z1: [B, D] - Representation from view 1
            z2: [B, D] - Representation from view 2

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        sim_loss = F.mse_loss(z1, z2)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = (
            self.inv_coeff * sim_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }
