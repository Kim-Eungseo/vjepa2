# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import json
import os
import random
from logging import getLogger

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.bg_compositor import BackgroundCompositor

logger = getLogger()


class ManiSkillPairDataset(Dataset):
    """
    ManiSkill DR 데이터셋에서 task / domain positive pair를 동시에 로드하는 Dataset.

    디렉토리 구조:
        root/
          configs.json            (optional, camera sampling weights)
          ep000/
            anchor/
              obs/frame_000.png   (H=512, W=512, RGB)
              mask/mask_000.png   (H=512, W=512, grayscale: 255=floor)
            cam_007/obs/frame_*.png
            ...
          ep001/
            ...

    __getitem__ 반환값 (dict):
        {
            "task": [(clip_A, clip_B), (clip_C, clip_D)],
            "dom":  [(clip_E, clip_F), (clip_G, clip_H)],
        }

        task pair: same segment, different domain
            clip_A, clip_C  ← anchor + 랜덤 배경 합성
            clip_B, clip_D  ← cam_XXX pre-rendered (다른 카메라)

        dom pair: different segment, same domain
            clip_E, clip_G  ← ep_i / cam_K  (같은 카메라 ID)
            clip_F, clip_H  ← ep_j / cam_K  (다른 에피소드)
    """

    def __init__(
        self,
        root_paths: list,
        fpc: int,
        transform,
        texture_dir: str,
        p_bg_augment: float = 0.8,
    ):
        if isinstance(root_paths, str):
            root_paths = [root_paths]

        self.fpc = fpc
        self.transform = transform
        self.compositor = BackgroundCompositor(texture_dir=texture_dir, p_augment=p_bg_augment)

        # ------------------------------------------------------------------ #
        # 1. 에피소드 인덱스 구축
        self._index = []
        self._cumulative_windows = []
        total = 0

        for root in root_paths:
            cam_weights_map = self._load_cam_weights(root)
            ep_dirs = sorted(
                d for d in os.listdir(root)
                if d.startswith("ep") and os.path.isdir(os.path.join(root, d))
            )
            for ep_name in ep_dirs:
                ep_dir = os.path.join(root, ep_name)
                num_frames = self._count_anchor_frames(ep_dir)
                if num_frames < fpc:
                    continue
                cam_ids = self._find_cam_ids(ep_dir)
                if not cam_ids:
                    continue
                cam_weights = self._make_cam_weights(cam_ids, cam_weights_map)
                windows = num_frames - fpc + 1
                self._index.append({
                    "ep_dir": ep_dir,
                    "num_frames": num_frames,
                    "cam_ids": cam_ids,
                    "cam_weights": cam_weights,
                })
                total += windows
                self._cumulative_windows.append(total)

        if not self._index:
            raise RuntimeError(f"No valid episodes found in {root_paths}")

        # ------------------------------------------------------------------ #
        # 2. domain pair용 역인덱스: cam_id → [ep_entry, ...]
        self._cam_index: dict = {}
        for ep_entry in self._index:
            for cam_id in ep_entry["cam_ids"]:
                self._cam_index.setdefault(cam_id, []).append(ep_entry)

        # 2개 이상의 에피소드가 있는 cam_id만 domain pair로 사용 가능
        self._valid_dom_cams = [
            c for c, eps in self._cam_index.items() if len(eps) >= 2
        ]

        if not self._valid_dom_cams:
            logger.warning(
                "No camera ID shared by ≥2 episodes. "
                "Domain pairs will fall back to different windows of the same episode."
            )

        logger.info(
            f"ManiSkillPairDataset: {len(self._index)} episodes, "
            f"{total} sliding windows (fpc={fpc}), "
            f"{len(self._valid_dom_cams)} domain-eligible cam IDs"
        )

    # ---------------------------------------------------------------------- #
    # 인덱스 구축 헬퍼

    def _load_cam_weights(self, root: str) -> dict:
        cfg_path = os.path.join(root, "configs.json")
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("camera_sampling_weights", {})

    def _count_anchor_frames(self, ep_dir: str) -> int:
        obs_dir = os.path.join(ep_dir, "anchor", "obs")
        if not os.path.isdir(obs_dir):
            return 0
        return len([f for f in os.listdir(obs_dir) if f.endswith(".png")])

    def _find_cam_ids(self, ep_dir: str) -> list:
        cam_ids = []
        for name in os.listdir(ep_dir):
            if not name.startswith("cam_"):
                continue
            obs_dir = os.path.join(ep_dir, name, "obs")
            if os.path.isdir(obs_dir):
                cam_ids.append(name[len("cam_"):])
        return sorted(cam_ids)

    def _make_cam_weights(self, cam_ids: list, weights_map: dict) -> list:
        if not weights_map:
            n = len(cam_ids)
            return [1.0 / n] * n
        raw = [float(weights_map.get(cid, 1.0)) for cid in cam_ids]
        total = sum(raw)
        return [w / total for w in raw]

    # ---------------------------------------------------------------------- #
    # Dataset 인터페이스

    def __len__(self) -> int:
        return self._cumulative_windows[-1] if self._cumulative_windows else 0

    def __getitem__(self, idx: int) -> dict:
        # ---- task pairs: same segment, different domain ----
        ep_entry, window_start = self._idx_to_ep_window(idx)
        num_frames = ep_entry["num_frames"]
        frame_indices = [min(window_start + i, num_frames - 1) for i in range(self.fpc)]

        anchor_obs = os.path.join(ep_entry["ep_dir"], "anchor", "obs")
        anchor_mask = os.path.join(ep_entry["ep_dir"], "anchor", "mask")

        cam_id_1 = self._sample_cam_id(ep_entry)
        cam_id_2 = self._sample_cam_id(ep_entry)
        cam1_obs = os.path.join(ep_entry["ep_dir"], f"cam_{cam_id_1}", "obs")
        cam2_obs = os.path.join(ep_entry["ep_dir"], f"cam_{cam_id_2}", "obs")

        frames_A = self._load_clip(anchor_obs, anchor_mask, frame_indices)
        frames_B = self._load_clip(cam1_obs, None, frame_indices)
        frames_C = self._load_clip(anchor_obs, anchor_mask, frame_indices)
        frames_D = self._load_clip(cam2_obs, None, frame_indices)

        # 같은 pair 내: 동일 spatial crop
        seed_t1 = random.randint(0, 2 ** 31)
        seed_t2 = random.randint(0, 2 ** 31)
        clip_A = self._apply_transform(frames_A, seed_t1)
        clip_B = self._apply_transform(frames_B, seed_t1)
        clip_C = self._apply_transform(frames_C, seed_t2)
        clip_D = self._apply_transform(frames_D, seed_t2)

        task_pairs = [(clip_A, clip_B), (clip_C, clip_D)]

        # ---- domain pairs: different segment, same domain ----
        dom_pairs = [self._sample_dom_pair(), self._sample_dom_pair()]

        return {"task": task_pairs, "dom": dom_pairs}

    # ---------------------------------------------------------------------- #
    # 내부 헬퍼

    def _idx_to_ep_window(self, idx: int):
        ep_idx = bisect.bisect_right(self._cumulative_windows, idx)
        ep_idx = min(ep_idx, len(self._index) - 1)
        prev = self._cumulative_windows[ep_idx - 1] if ep_idx > 0 else 0
        window_start = idx - prev
        return self._index[ep_idx], window_start

    def _sample_cam_id(self, ep_entry: dict) -> str:
        return random.choices(ep_entry["cam_ids"], weights=ep_entry["cam_weights"], k=1)[0]

    def _sample_dom_pair(self):
        """
        Different segment, same domain (cam_id) → domain positive pair.

        같은 cam_id를 공유하는 다른 에피소드에서 임의의 윈도우를 하나씩 샘플링한다.
        cam_id를 공유하는 에피소드가 없으면, 동일 에피소드의 서로 다른 윈도우로 fallback.
        """
        if self._valid_dom_cams:
            cam_id = random.choice(self._valid_dom_cams)
            ep1, ep2 = random.sample(self._cam_index[cam_id], 2)

            w1 = random.randint(0, ep1["num_frames"] - self.fpc)
            w2 = random.randint(0, ep2["num_frames"] - self.fpc)
            fi1 = [min(w1 + i, ep1["num_frames"] - 1) for i in range(self.fpc)]
            fi2 = [min(w2 + i, ep2["num_frames"] - 1) for i in range(self.fpc)]

            obs1 = os.path.join(ep1["ep_dir"], f"cam_{cam_id}", "obs")
            obs2 = os.path.join(ep2["ep_dir"], f"cam_{cam_id}", "obs")
        else:
            # fallback: 같은 에피소드, 다른 윈도우
            ep = random.choice(self._index)
            cam_id = self._sample_cam_id(ep)
            obs1 = obs2 = os.path.join(ep["ep_dir"], f"cam_{cam_id}", "obs")

            max_start = ep["num_frames"] - self.fpc
            w1 = random.randint(0, max_start)
            w2 = random.randint(0, max_start)
            fi1 = [min(w1 + i, ep["num_frames"] - 1) for i in range(self.fpc)]
            fi2 = [min(w2 + i, ep["num_frames"] - 1) for i in range(self.fpc)]

        frames1 = self._load_clip(obs1, None, fi1)
        frames2 = self._load_clip(obs2, None, fi2)

        # domain pair는 서로 다른 scene이므로 독립적인 crop seed 사용
        clip1 = self._apply_transform(frames1, random.randint(0, 2 ** 31))
        clip2 = self._apply_transform(frames2, random.randint(0, 2 ** 31))
        return (clip1, clip2)

    def _load_clip(self, obs_dir: str, mask_dir, frame_indices: list) -> np.ndarray:
        frames = []
        for idx in frame_indices:
            frame = imageio.imread(os.path.join(obs_dir, f"frame_{idx:03d}.png"))
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            if mask_dir is not None:
                mask = imageio.imread(os.path.join(mask_dir, f"mask_{idx:03d}.png"))
                frame = self.compositor.apply(frame, mask)
            frames.append(frame)
        return np.stack(frames, axis=0)  # (fpc, H, W, 3)

    def _apply_transform(self, frames: np.ndarray, seed: int) -> torch.Tensor:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        return self.transform(frames)
