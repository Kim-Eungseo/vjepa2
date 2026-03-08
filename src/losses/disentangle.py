# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class CrossCorrelationSepLoss(nn.Module):
    """Cross-correlation separation loss between two representation subspaces.

    Computes the cross-correlation matrix between z_task and z_dom (after
    batch mean-centering) and penalises all entries toward zero.  This forces
    the two heads to encode statistically independent information.

    Inspired by Barlow Twins (Zbontar et al., ICML 2021), but targets the
    **zero matrix** instead of identity — because the two heads should share
    *no* mutual information rather than encode the *same* information.

    L_sep = sum_{i,j} C_{ij}^2

    where C_{ij} = corr(z_task[:, i], z_dom[:, j]) over the batch.
    """

    def forward(self, z_task: torch.Tensor, z_dom: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_task: [B, d] task-biased representations
            z_dom:  [B, d] domain-biased representations

        Returns:
            Scalar loss (sum of squared cross-correlations).
        """
        # Batch mean-center
        z_task = z_task - z_task.mean(dim=0, keepdim=True)
        z_dom = z_dom - z_dom.mean(dim=0, keepdim=True)

        # Normalise each feature dimension by its std
        z_task_std = torch.sqrt(z_task.var(dim=0) + 1e-4)
        z_dom_std = torch.sqrt(z_dom.var(dim=0) + 1e-4)

        z_task = z_task / z_task_std
        z_dom = z_dom / z_dom_std

        # Cross-correlation matrix [d, d]
        batch_size = z_task.shape[0]
        c = (z_task.T @ z_dom) / batch_size  # [d, d]

        # All entries should be zero
        loss = c.pow(2).sum()
        return loss
