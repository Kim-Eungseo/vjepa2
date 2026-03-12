# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import cv2
import numpy as np

TEXTURE_FILES = [
    "bricks.jpg",
    "cliffdesert.jpg",
    "cobblestone.png",
    "fabricclothes.jpg",
    "fabricpattern.png",
    "fabricsuedefine.jpg",
    "fabrictarpplastic.png",
    "metal.png",
]


class BackgroundCompositor:
    """
    anchor/obs/frame_*.png + anchor/mask/mask_*.png 으로부터 배경 합성된 프레임을 반환.

    Args:
        texture_dir: 텍스처 이미지 파일들이 있는 디렉토리
        p_augment:   배경을 실제로 바꿀 확률 (0.0=항상 원본, 1.0=항상 합성)
    """

    def __init__(self, texture_dir: str, p_augment: float = 0.8):
        self.texture_dir = texture_dir
        self.p_augment = p_augment
        self._cache: dict = {}  # key → resized RGB ndarray

    def _load_texture(self, filename: str, h: int, w: int) -> np.ndarray:
        key = f"{filename}_{h}_{w}"
        if key not in self._cache:
            path = os.path.join(self.texture_dir, filename)
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Texture not found: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            self._cache[key] = img
        return self._cache[key]

    def apply(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        texture_filename: str | None = None,
    ) -> np.ndarray:
        """
        Args:
            frame:             (H, W, 3) uint8 RGB
            mask:              (H, W)    uint8, 255 = floor pixel
            texture_filename:  None이면 랜덤 선택

        Returns:
            합성된 (H, W, 3) uint8 RGB
        """
        if random.random() > self.p_augment:
            return frame

        h, w = frame.shape[:2]
        if texture_filename is None:
            texture_filename = random.choice(TEXTURE_FILES)

        texture = self._load_texture(texture_filename, h, w)
        result = frame.copy()
        result[mask == 255] = texture[mask == 255]
        return result
