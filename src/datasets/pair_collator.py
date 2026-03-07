# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class PairCollator(object):
    """
    Collator for positive-pair contrastive learning.

    Expects each sample from the dataset to contain at least 2 clips
    (buffer = [clip1, clip2, ...]), e.g. via num_clips=2 in VideoDataset.

    Returns a list of (clip1_batch, clip2_batch) tuples, one per unique
    frames_per_clip value in the batch.

    clip1_batch, clip2_batch: [B, C, T, H, W]
    """

    def __init__(self, dataset_fpcs):
        self.dataset_fpcs = dataset_fpcs

    def __call__(self, batch):
        # Group samples by frames_per_clip
        filtered_batches = {fpc: [] for fpc in self.dataset_fpcs}
        for sample in batch:
            # sample = (buffer, label, clip_indices)
            # clip_indices[-1] has length = frames_per_clip
            fpc = len(sample[-1][-1])
            filtered_batches[fpc].append(sample)

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            if len(fpc_batch) == 0:
                continue

            # Stack clip1 and clip2 separately across batch
            clip1_list = [sample[0][0] for sample in fpc_batch]  # first clip per sample
            clip2_list = [sample[0][1] for sample in fpc_batch]  # second clip per sample

            clip1_batch = torch.stack(clip1_list)  # [B, C, T, H, W]
            clip2_batch = torch.stack(clip2_list)  # [B, C, T, H, W]

            fpc_collations.append((clip1_batch, clip2_batch))

        return fpc_collations
