# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
from logging import getLogger

import numpy as np
import torch
import torchvision.io
from torch.utils.data import Dataset

logger = getLogger()


def make_episode_dataset(
    data_paths,
    batch_size,
    frames_per_clip=16,
    dataset_fpcs=None,
    num_clips=2,
    transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    view_types=None,
):
    """Create EpisodeDataset with DataLoader and DistributedSampler.

    Args:
        data_paths: list of root dirs (e.g. ["data/LiftPegUpright-v1"])
        view_types: list of view prefixes to use as augmentation sources.
            Default: ["cam", "lighting", "bg"] (all view types).
    """
    dataset = EpisodeDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        num_clips=num_clips,
        transform=transform,
        view_types=view_types,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info(f"EpisodeDataset created: {len(dataset)} samples, {len(dataset.view_dirs)} views per episode")
    return dataset, data_loader, dist_sampler


class EpisodeDataset(Dataset):
    """Dataset for robotics episode data with multi-view positive pairs.

    Data layout expected:
        root/
          episode_X/
            <view_dir>/          # e.g. anchor, cam_000, bg_001, lighting_003
              meta.json
              observation/
                frame_000.png
                frame_001.png
                ...

    For each sample, two different views of the same temporal clip are returned
    as a positive pair, compatible with PairCollator.
    """

    def __init__(
        self,
        data_paths,
        frames_per_clip=16,
        dataset_fpcs=None,
        num_clips=2,
        transform=None,
        view_types=None,
    ):
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.frames_per_clip = frames_per_clip
        self.dataset_fpcs = dataset_fpcs or [frames_per_clip]
        self.num_clips = num_clips
        self.transform = transform
        self.view_types = view_types or ["cam", "lighting", "bg"]

        # Discover episodes and views
        self.episodes = []  # list of (episode_dir, num_frames)
        self.view_dirs = []  # list of view dir names (shared across episodes)

        for root in data_paths:
            episode_dirs = sorted(
                [d for d in os.listdir(root) if d.startswith("episode_") and os.path.isdir(os.path.join(root, d))]
            )
            if not episode_dirs:
                logger.warning(f"No episodes found in {root}")
                continue

            # Discover view dirs from first episode
            if not self.view_dirs:
                first_ep = os.path.join(root, episode_dirs[0])
                self.view_dirs = self._discover_views(first_ep)
                logger.info(f"Discovered {len(self.view_dirs)} views: {self.view_dirs[:5]}...")

            for ep_name in episode_dirs:
                ep_dir = os.path.join(root, ep_name)
                # Get num_frames from anchor meta or first available view
                num_frames = self._get_num_frames(ep_dir)
                if num_frames > 0:
                    self.episodes.append((ep_dir, num_frames))

        if not self.episodes:
            raise RuntimeError(f"No valid episodes found in {data_paths}")
        if len(self.view_dirs) < 2:
            raise RuntimeError(f"Need at least 2 views for pairs, found {len(self.view_dirs)}")

        logger.info(f"Found {len(self.episodes)} episodes, {len(self.view_dirs)} views")

    def _discover_views(self, episode_dir):
        """Find all view directories matching view_types."""
        views = []
        for name in sorted(os.listdir(episode_dir)):
            subdir = os.path.join(episode_dir, name)
            if not os.path.isdir(subdir):
                continue
            if any(name.startswith(prefix) for prefix in self.view_types + ["anchor"]):
                obs_dir = os.path.join(subdir, "observation")
                if os.path.isdir(obs_dir):
                    views.append(name)
        return views

    def _get_num_frames(self, episode_dir):
        """Get number of frames from meta.json or by counting files."""
        for view in ["anchor"] + self.view_dirs[:1]:
            meta_path = os.path.join(episode_dir, view, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                return meta.get("num_frames", 0)
        # Fallback: count frames
        for view in self.view_dirs:
            obs_dir = os.path.join(episode_dir, view, "observation")
            if os.path.isdir(obs_dir):
                return len([f for f in os.listdir(obs_dir) if f.endswith(".png")])
        return 0

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        ep_dir, num_frames = self.episodes[index]
        fpc = self.dataset_fpcs[0] if self.dataset_fpcs else self.frames_per_clip

        # Sample a random temporal window
        max_start = max(0, num_frames - fpc)
        start_idx = random.randint(0, max_start)
        frame_indices = np.arange(start_idx, min(start_idx + fpc, num_frames))
        # Pad if not enough frames
        if len(frame_indices) < fpc:
            pad = np.full(fpc - len(frame_indices), frame_indices[-1])
            frame_indices = np.concatenate([frame_indices, pad])

        # Pick 2 different views for positive pair
        view1, view2 = random.sample(self.view_dirs, 2)

        clip1 = self._load_clip(ep_dir, view1, frame_indices)
        clip2 = self._load_clip(ep_dir, view2, frame_indices)

        # Apply transforms (clip is [T, H, W, C] numpy → transform expects this)
        if self.transform is not None:
            clip1 = self.transform(clip1)
            clip2 = self.transform(clip2)

        label = 0
        clip_indices = [frame_indices, frame_indices]
        return [clip1, clip2], label, clip_indices

    def _load_clip(self, episode_dir, view_name, frame_indices):
        """Load a sequence of frames as [T, H, W, C] uint8 numpy array."""
        obs_dir = os.path.join(episode_dir, view_name, "observation")
        frames = []
        for idx in frame_indices:
            frame_path = os.path.join(obs_dir, f"frame_{idx:03d}.png")
            if not os.path.exists(frame_path):
                # Try finding closest available frame
                frame_path = self._find_closest_frame(obs_dir, idx)
            img = torchvision.io.read_image(frame_path, mode=torchvision.io.ImageReadMode.RGB)
            # img: [3, H, W] → [H, W, 3]
            frames.append(img.permute(1, 2, 0).numpy())
        return np.stack(frames)  # [T, H, W, C]

    def _find_closest_frame(self, obs_dir, target_idx):
        """Find the closest available frame file."""
        files = sorted(os.listdir(obs_dir))
        png_files = [f for f in files if f.endswith(".png")]
        if not png_files:
            raise FileNotFoundError(f"No frames in {obs_dir}")
        # Extract indices and find closest
        indices = [int(f.split("_")[1].split(".")[0]) for f in png_files]
        closest = min(indices, key=lambda x: abs(x - target_idx))
        return os.path.join(obs_dir, f"frame_{closest:03d}.png")
