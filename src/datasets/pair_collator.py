# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class PairCollator(object):
    """
    Collator for positive-pair contrastive learning.

    두 가지 입력 포맷을 지원한다:

    1. VideoDataset / EpisodeDataset 포맷 (기존):
       sample = (buffer, label, clip_indices)
       buffer = [clip1, clip2, ...]
       clip_indices[-1] 길이로 fpc를 판별해 그룹핑.

    2. ManiSkillPairDataset 포맷 (신규):
       sample = [(clip_A, clip_B), (clip_C, clip_D), ...]  # list of (view1, view2) pairs
       clip shape: (C, T, H, W) float32 tensor

    두 경우 모두 반환값:
        list of (clip1_batch, clip2_batch)  # [B, C, T, H, W] × 2, per pair
    """

    def __init__(self, dataset_fpcs):
        self.dataset_fpcs = dataset_fpcs

    def __call__(self, batch):
        # ManiSkillPairDataset: batch[0] is a dict {"task": [...], "dom": [...]}
        if isinstance(batch[0], dict):
            return self._collate_maniskill(batch)
        return self._collate_video_dataset(batch)

    def _collate_maniskill(self, batch):
        """
        ManiSkillPairDataset 포맷 collation.

        batch[i] = {"task": [(clip_A, clip_B), ...], "dom": [(clip_E, clip_F), ...]}
        반환:
        {
            "task": [(stack_A, stack_B), ...],   # (B, C, T, H, W) × 2
            "dom":  [(stack_E, stack_F), ...],   # (B, C, T, H, W) × 2
        }
        """
        def stack_pairs(key):
            n_pairs = len(batch[0][key])
            result = []
            for pair_idx in range(n_pairs):
                v1 = torch.stack([item[key][pair_idx][0] for item in batch])
                v2 = torch.stack([item[key][pair_idx][1] for item in batch])
                result.append((v1, v2))
            return result

        return {"task": stack_pairs("task"), "dom": stack_pairs("dom")}

    def _collate_video_dataset(self, batch):
        """
        VideoDataset / EpisodeDataset 포맷 collation (기존 동작).

        sample = (buffer, label, clip_indices)
        """
        filtered_batches = {fpc: [] for fpc in self.dataset_fpcs}
        for sample in batch:
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
