# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler


class ProjectionHead(nn.Module):
    """AttentivePooler followed by a linear projection.

    Used as task-biased or domain-biased head in the disentanglement encoder.
    Pooler compresses [B, N, D] token sequences to [B, D], then the linear
    layer projects to the target dimension.
    """

    def __init__(
        self,
        embed_dim,
        proj_dim,
        num_heads,
        pooler_depth=1,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=pooler_depth,
        )
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, h):
        """
        Args:
            h: [B, N, D] encoder token representations
        Returns:
            z: [B, proj_dim] projected representation
        """
        z = self.pooler(h).squeeze(1)  # [B, D]
        z = self.proj(z)                # [B, proj_dim]
        return z
