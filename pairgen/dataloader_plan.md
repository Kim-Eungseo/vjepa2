# ManiSkill Pair DataLoader 구현 계획

## 개요

ManiSkill DR 데이터셋에서 positive pair를 로드하는 DataLoader를 구현한다.
제공된 `post_train.py`의 학습 루프와 완전히 호환되어야 한다.

**핵심 요구사항:**
- `dataset_fpcs: [16, 16]` → 에피소드당 2쌍의 positive pair
- pair = (anchor+온라인 배경 합성, cam_XXX pre-rendered)
- `PairCollator`가 배치로 묶어 `for clip1_batch, clip2_batch in sample` 루프와 호환
- 배경 DR은 오프라인 렌더링 없이 온라인 합성으로 처리 (80× 저장 절감)

---

## 전제: 데이터셋 디렉토리 구조

```
/data/datasets/{task}/           e.g. PickCube-v1/
├── configs.json                 generate_configs.py 출력 (카메라 가중치 포함)
├── ep000/
│   ├── anchor/
│   │   ├── obs/
│   │   │   ├── frame_000.png    (H=512, W=512, RGB)
│   │   │   ├── frame_001.png
│   │   │   └── ...
│   │   └── mask/                ← replay_anchor.py 수정으로 추가
│   │       ├── mask_000.png     (H=512, W=512, grayscale: 255=floor)
│   │       ├── mask_001.png
│   │       └── ...
│   ├── cam_000/obs/frame_*.png
│   ├── cam_007/obs/frame_*.png
│   └── ...
├── ep001/
│   └── ...
```

**주의:** `cam_*/` 디렉토리는 `replay_dr.py --dr-type camera`로 사전 생성 필요.
`anchor/mask/`는 아래 1단계 수정 후 `replay_anchor.py` 재실행으로 생성.

---

## 구현 순서

```
Step 1. replay_anchor.py 수정    → mask 저장 기능 추가
Step 2. bg_compositor.py         → 온라인 배경 합성 유틸
Step 3. maniskill_pair_dataset.py → Dataset 클래스
Step 4. pair_collator.py          → Collator
Step 5. data_manager.py           → init_data() 진입점
Step 6. config 수정               → dataset_type / dataset_fpcs
```

---

## Step 1. `replay_anchor.py` 수정

### 변경 목적
앵커 렌더링 시 바닥(floor) 세그멘테이션 마스크를 `anchor/mask/mask_NNN.png`로 함께 저장한다.

### 변경 위치: `replay_worker()`

```python
# 기존 env_kwargs
env_kwargs["render_mode"] = "rgb_array"

# 변경: segmentation 추가
env_kwargs["render_mode"] = "rgb_array"
env_kwargs["obs_mode"] = "rgb+segmentation"
env_kwargs["sensor_configs"] = {
    "base_camera": {
        "pose": anchor_pose,
        "width": 512,
        "height": 512,
        "fov": 1,
    }
}
```

### 변경 위치: `replay_episode()`

```python
def replay_episode(env, h5_file, episode, anchor_info, output_dir):
    episode_id = episode["episode_id"]
    traj_id = f"traj_{episode_id}"
    anchor_dir = os.path.join(output_dir, f"ep{episode_id:03d}", "anchor")
    obs_dir  = os.path.join(anchor_dir, "obs")
    mask_dir = os.path.join(anchor_dir, "mask")   # ← 추가
    os.makedirs(obs_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)           # ← 추가

    env.reset(**reset_kwargs)
    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    env.unwrapped.set_state_dict(ori_env_states[0])

    # floor actor의 seg ID 추출 (reset 후 scene 접근)
    scene = env.unwrapped.scene
    ground_actor = scene.actors["ground"]
    floor_seg_ids = ground_actor.per_scene_id.to(dtype=torch.int16)

    ori_actions = h5_file[traj_id]["actions"][:]

    def save_frame_and_mask(frame_idx):
        # RGB 렌더링
        frame = env.render().cpu().numpy()[0]
        imageio.imsave(os.path.join(obs_dir, f"frame_{frame_idx:03d}.png"), frame)

        # Segmentation 마스크 추출
        obs = env.unwrapped.get_obs()
        seg = obs["sensor_data"]["base_camera"]["segmentation"]  # (1, H, W, 1)
        mask = torch.isin(seg[0, :, :, 0], floor_seg_ids.to(seg.device))
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        imageio.imsave(os.path.join(mask_dir, f"mask_{frame_idx:03d}.png"), mask_np)

    save_frame_and_mask(0)

    for i, a in enumerate(ori_actions):
        env.step(a)
        env.unwrapped.set_state_dict(ori_env_states[i + 1])
        save_frame_and_mask(i + 1)

    return len(ori_actions) + 1
```

### 추가 import
```python
import torch
```

---

## Step 2. `src/datasets/bg_compositor.py`

### 역할
- 텍스처 이미지를 lazy하게 로드 (캐시)
- floor mask 기반으로 배경 픽셀을 랜덤 텍스처로 교체

### 전체 코드 설계

```python
import os
import random
import numpy as np
import cv2

TEXTURE_FILES = [
    "bricks.jpg", "cliffdesert.jpg", "cobblestone.png",
    "fabricclothes.jpg", "fabricpattern.png", "fabricsuedefine.jpg",
    "fabrictarpplastic.png", "metal.png",
]

class BackgroundCompositor:
    """
    anchor/obs/frame_*.png + anchor/mask/mask_*.png → 배경 합성된 프레임 반환.

    Args:
        texture_dir: 텍스처 이미지 파일들이 있는 디렉토리
        p_augment:   배경을 실제로 바꿀 확률 (0.0이면 항상 원본, 1.0이면 항상 합성)
        target_size: 텍스처를 리사이즈할 (H, W). 기본값은 프레임 크기에 맞춤.
    """

    def __init__(self, texture_dir: str, p_augment: float = 0.8,
                 target_size: tuple[int, int] | None = None):
        self.texture_dir = texture_dir
        self.p_augment = p_augment
        self.target_size = target_size
        self._cache: dict[str, np.ndarray] = {}  # filename → resized RGB ndarray

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

    def apply(self, frame: np.ndarray, mask: np.ndarray,
              texture_filename: str | None = None) -> np.ndarray:
        """
        Args:
            frame:             (H, W, 3) uint8 RGB
            mask:              (H, W)    uint8, 255 = floor pixel
            texture_filename:  None이면 랜덤 선택

        Returns:
            합성된 (H, W, 3) uint8 RGB
        """
        if random.random() > self.p_augment:
            return frame  # 원본 반환

        h, w = frame.shape[:2]
        if texture_filename is None:
            texture_filename = random.choice(TEXTURE_FILES)

        texture = self._load_texture(texture_filename, h, w)
        result = frame.copy()
        result[mask == 255] = texture[mask == 255]
        return result
```

---

## Step 3. `src/datasets/maniskill_pair_dataset.py`

### 역할
- 에피소드 디렉토리를 스캔해 인덱스 구축
- `__getitem__`에서 positive pair 2쌍 반환
- 각 쌍 = (anchor+bg_aug, cam_XXX) 구성

### 인덱스 구조
```python
# self._index: list of dict
{
    "ep_dir":    "/data/datasets/PickCube-v1/ep000",
    "num_frames": 87,           # anchor/obs/ 내 frame 개수
    "cam_ids":   ["007", "042", "103", ...],  # 가용 cam_* ID 목록
}
```

### `__len__` 계산
```python
# 에피소드당 슬라이딩 윈도우 수
windows_per_ep = max(0, num_frames - fpc + 1)
total = sum(windows_per_ep for each episode)
```

### `__getitem__` 상세 흐름

```
idx 입력
  │
  ├─ 1. idx → (ep_entry, window_start) 변환
  │         (누적 윈도우 수로 이진 탐색)
  │
  ├─ 2. frame 인덱스 리스트 생성
  │         frame_indices = [window_start, window_start+1, ..., window_start+fpc-1]
  │
  ├─ 3. pair 1 구성
  │     view_A: anchor + BackgroundCompositor (texture_1 랜덤 선택)
  │     view_B: cam_X  (configs.json 가중치 기반 랜덤 선택)
  │
  ├─ 4. pair 2 구성
  │     view_C: anchor + BackgroundCompositor (texture_2 랜덤 선택, texture_1과 독립)
  │     view_D: cam_Y  (view_B와 독립적으로 랜덤 선택, 같은 cam 가능)
  │
  ├─ 5. 각 view의 프레임 로드 (_load_clip)
  │     → (fpc, H, W, 3) ndarray
  │
  ├─ 6. transform 적용 (동일 pair 내 두 view에 같은 spatial crop 적용)
  │     → (C, fpc, H, W) float32 tensor   [vjepa 포맷]
  │
  └─ 7. 반환
        [(clip_A, clip_B),   # pair 1
         (clip_C, clip_D)]   # pair 2
```

### transform 적용 방식
같은 pair 내 두 view는 **동일한 random crop 파라미터**를 공유한다.
(위치 정보가 보존되어야 task head가 spatial correspondence를 학습 가능)

```python
# pair 내 두 view에 동일 seed로 transform
seed = random.randint(0, 2**32)
clip_A = apply_transform(frames_A, transform, seed=seed)
clip_B = apply_transform(frames_B, transform, seed=seed)
```

### `_load_clip()` 상세

```python
def _load_clip(self, obs_dir, mask_dir, frame_indices, compositor):
    """
    Args:
        obs_dir:       anchor/obs/ 또는 cam_XXX/obs/ 경로
        mask_dir:      anchor/mask/ 경로 (None이면 합성 안함)
        frame_indices: [i0, i1, ..., i_{fpc-1}]
        compositor:    BackgroundCompositor 인스턴스 (mask_dir=None이면 None)

    Returns:
        frames: (fpc, H, W, 3) ndarray uint8
    """
    frames = []
    for idx in frame_indices:
        frame = imageio.imread(os.path.join(obs_dir, f"frame_{idx:03d}.png"))
        # anchor view이면 배경 합성
        if compositor is not None and mask_dir is not None:
            mask = imageio.imread(os.path.join(mask_dir, f"mask_{idx:03d}.png"))
            frame = compositor.apply(frame, mask)
        frames.append(frame)
    return np.stack(frames, axis=0)  # (fpc, H, W, 3)
```

### 카메라 configs 가중치 샘플링

`configs.json`의 `camera_sampling_weights`를 그대로 활용:

```python
def _sample_cam_id(self, rng) -> str:
    """configs.json의 앵커 편향 가중치로 cam ID 샘플링."""
    ids = self._cam_ids          # ["0", "1", ..., "128"]
    weights = self._cam_weights  # 정규화된 float 리스트
    return rng.choice(ids, p=weights)
```

단, 해당 에피소드에 실제로 존재하는 `cam_*/` 디렉토리만 후보로 사용.

### 전체 클래스 시그니처

```python
class ManiSkillPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_paths: list[str],     # 데이터셋 루트 디렉토리 목록
        fpc: int,                   # frames per clip (e.g. 16)
        transform,                  # make_transforms() 반환값
        texture_dir: str,           # 텍스처 이미지 디렉토리
        p_bg_augment: float = 0.8,  # 배경 변경 확률
    )
```

---

## Step 4. `src/datasets/pair_collator.py`

### 역할
`Dataset.__getitem__`이 반환하는 `[(clip_A, clip_B), (clip_C, clip_D)]` 리스트를
배치 단위로 묶어 학습 루프가 기대하는 포맷으로 변환.

### 입출력

```
입력 (batch, 길이 B):
  [
    [(clip_A_0, clip_B_0), (clip_C_0, clip_D_0)],   # sample 0
    [(clip_A_1, clip_B_1), (clip_C_1, clip_D_1)],   # sample 1
    ...
  ]
  clip shape: (C, fpc, H, W) float32 tensor

출력 (학습 루프가 기대하는 포맷):
  [
    (clip1_batch_pair1, clip2_batch_pair1),   # (B, C, fpc, H, W) × 2
    (clip1_batch_pair2, clip2_batch_pair2),   # (B, C, fpc, H, W) × 2
  ]

학습 루프:
  for clip1_batch, clip2_batch in sample:    # ← 2번 반복 (2쌍)
      h1 = encoder(clip1_batch)
      h2 = encoder(clip2_batch)
```

### 구현

```python
class PairCollator:
    def __init__(self, dataset_fpcs: list[int]):
        self.n_pairs = len(dataset_fpcs)   # 2

    def __call__(self, batch: list) -> list[tuple]:
        # batch[i] = [(v1A, v2A), (v1B, v2B)]
        result = []
        for pair_idx in range(self.n_pairs):
            views1 = torch.stack([item[pair_idx][0] for item in batch])  # (B, C, T, H, W)
            views2 = torch.stack([item[pair_idx][1] for item in batch])  # (B, C, T, H, W)
            result.append((views1, views2))
        return result
```

---

## Step 5. `src/datasets/data_manager.py`

### 역할
`init_data()` 함수 하나를 제공. `dataset_type`에 따라 적절한 Dataset을 생성하고
`DataLoader`로 감싸서 반환.

### 함수 시그니처 (post_train.py 호출과 일치)

```python
def init_data(
    data: str,                          # "maniskill_pair"
    root_path: list[str],               # dataset 디렉토리 목록
    batch_size: int,
    training: bool,
    dataset_fpcs: list[int],            # [16, 16]
    fps: float | None,
    num_clips: int,                     # 2 (사용 안함, pair 개수는 fpcs 길이)
    transform,
    rank: int,
    world_size: int,
    datasets_weights: list | None,
    persistent_workers: bool,
    collator,                           # PairCollator 인스턴스
    num_workers: int,
    pin_mem: bool,
    log_dir: str | None,
    texture_dir: str = "",              # maniskill_pair 전용
    p_bg_augment: float = 0.8,         # maniskill_pair 전용
) -> tuple[DataLoader, DistributedSampler]:
```

### 분기 로직

```python
if data == "maniskill_pair":
    dataset = ManiSkillPairDataset(
        root_paths=root_path,
        fpc=max(dataset_fpcs),
        transform=transform,
        texture_dir=texture_dir,
        p_bg_augment=p_bg_augment,
    )
else:
    raise NotImplementedError(f"Unknown dataset type: {data}")

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=training,
    drop_last=True,
)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=persistent_workers and num_workers > 0,
    collate_fn=collator,
    drop_last=True,
)
return loader, sampler
```

---

## Step 6. Config 변경

```yaml
# 변경 전
data:
  dataset_type: VideoDataset
  datasets:
  - /your_data_path/dataset_train_paths.csv
  dataset_fpcs:
  - 16

# 변경 후
data:
  dataset_type: maniskill_pair
  datasets:
  - /data/datasets/PickCube-v1
  - /data/datasets/LiftPegUpright-v1
  - /data/datasets/StackCube-v1
  dataset_fpcs:
  - 16
  - 16                              # 2쌍
  texture_dir: /path/to/texture-images
  p_bg_augment: 0.8
```

`post_train.py`에서 `texture_dir`와 `p_bg_augment`를 `cfgs_data`에서 읽어서
`init_data()`에 넘겨야 함:

```python
texture_dir  = cfgs_data.get("texture_dir", "")
p_bg_augment = cfgs_data.get("p_bg_augment", 0.8)

(unsupervised_loader, unsupervised_sampler) = init_data(
    ...
    texture_dir=texture_dir,
    p_bg_augment=p_bg_augment,
)
```

---

## 파일 목록 요약

| 파일 | 동작 |
|------|------|
| `maniskill_copy/maniskill/replay_anchor.py` | 수정: mask 저장 추가 |
| `src/datasets/__init__.py` | 신규 (빈 파일) |
| `src/datasets/bg_compositor.py` | 신규 |
| `src/datasets/maniskill_pair_dataset.py` | 신규 |
| `src/datasets/pair_collator.py` | 신규 |
| `src/datasets/data_manager.py` | 신규 |
| `configs/post_train.yaml` | 수정: dataset_type, fpcs, texture_dir |
| `app/post_train.py` | 수정: texture_dir, p_bg_augment 파라미터 전달 |

---

## 데이터 흐름 최종 확인

```
disk
  anchor/obs/frame_000.png  (512×512 RGB)
  anchor/mask/mask_000.png  (512×512 grayscale)
  cam_042/obs/frame_000.png (512×512 RGB)
      │
      ▼
ManiSkillPairDataset.__getitem__()
  → BackgroundCompositor.apply(frame, mask)   # CPU, ~0.2ms/frame
  → transform (RandomResizedCrop → 256×256, flip)
  → (C=3, T=16, H=256, W=256) float32 tensor × 4 views
  → [(view_A, view_B), (view_C, view_D)]
      │
      ▼
PairCollator.__call__(batch)
  → [(clip1_batch, clip2_batch),   # (24, 3, 16, 256, 256) × 2
     (clip1_batch, clip2_batch)]   # (24, 3, 16, 256, 256) × 2
      │
      ▼
post_train.py 학습 루프
  for clip1_batch, clip2_batch in sample:    # 2번 반복
      h1 = encoder(clip1_batch)   # (24, N, D)
      h2 = encoder(clip2_batch)
      z_task1 = task_head(h1)     # (24, 256)
      z_task2 = task_head(h2)
      loss = VICReg(z_task1, z_task2) + ...
```
