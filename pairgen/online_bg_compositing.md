# 배경 DR 온라인 합성 (Online Background Compositing)

## 문제

현재 파이프라인에서 배경 Domain Randomization은 **80가지 텍스처 조합**을 에피소드별로 전부 오프라인 렌더링해서 저장한다.

```
ep000/
  anchor/obs/         frame_000.png ~ frame_NNN.png   ← 1×T 장
  bg_000/obs/         frame_000.png ~ frame_NNN.png   ← 1×T 장
  bg_001/obs/         ...                             ← 1×T 장
  ...
  bg_079/obs/         frame_000.png ~ frame_NNN.png   ← 1×T 장
                                             합계: 81×T 장
```

배경이 달라도 **로봇과 오브젝트의 외형은 동일**하다. 바뀌는 것은 바닥/테이블 픽셀뿐이다.
이미 green-screen 합성 방식을 쓰고 있기 때문에, 이 작업을 오프라인에서 할 이유가 없다.

---

## 해결책

앵커 렌더링 시 **RGB 프레임과 세그멘테이션 마스크를 함께 저장**하고,
배경 DR은 학습 시 on-the-fly로 텍스처를 합성한다.

```
ep000/
  anchor/obs/         frame_000.png ~ frame_NNN.png   ← T 장 (RGB)
  anchor/mask/        mask_000.png  ~ mask_NNN.png    ← T 장 (floor mask)
                                             합계: 2×T 장

학습 시: anchor RGB + mask + 랜덤 텍스처 이미지 → 합성
```

저장량: **81T → 2T** (약 40× 감소)

---

## 변경 사항

### 1. `replay_anchor.py` 수정

앵커 렌더링 시 바닥 세그멘테이션 마스크를 함께 저장한다.

```python
# env 설정에 segmentation 추가
env_kwargs["obs_mode"] = "rgb+segmentation"
env_kwargs["sensor_configs"] = {
    "base_camera": {
        "pose": anchor_pose, "width": 512, "height": 512, "fov": 1,
    }
}

# 마스크 저장 디렉토리
mask_dir = os.path.join(anchor_dir, "mask")
os.makedirs(mask_dir, exist_ok=True)

# 바닥 actor의 seg ID 추출
ground_actor = env.unwrapped.scene.actors["ground"]
floor_seg_ids = ground_actor.per_scene_id.to(dtype=torch.int16)

# 렌더링 루프에서
obs = env.unwrapped.get_obs()
seg = obs["sensor_data"]["base_camera"]["segmentation"]  # (1, H, W, 1)
mask = torch.isin(seg[0, :, :, 0], floor_seg_ids.to(seg.device))
mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
imageio.imsave(os.path.join(mask_dir, f"mask_{i:03d}.png"), mask_np)
```

### 2. 학습 DataLoader에서 온라인 합성

```python
import cv2
import numpy as np
import random
from pathlib import Path

TEXTURE_DIR = "/path/to/texture-images"
TEXTURES = [
    "bricks.jpg", "cliffdesert.jpg", "cobblestone.png",
    "fabricclothes.jpg", "fabricpattern.png", "fabricsuedefine.jpg",
    "fabrictarpplastic.png", "metal.png",
]

def composite_bg(frame: np.ndarray, mask: np.ndarray, texture_path: str) -> np.ndarray:
    """mask=255인 픽셀을 texture로 교체한다."""
    h, w = frame.shape[:2]
    tex = cv2.imread(texture_path)
    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    tex = cv2.resize(tex, (w, h))
    result = frame.copy()
    result[mask == 255] = tex[mask == 255]
    return result

def load_frame_with_bg_aug(ep_dir: str, frame_idx: int) -> np.ndarray:
    """앵커 프레임을 로드하고 랜덤 배경을 합성한다."""
    frame = imageio.imread(f"{ep_dir}/anchor/obs/frame_{frame_idx:03d}.png")
    mask  = imageio.imread(f"{ep_dir}/anchor/mask/mask_{frame_idx:03d}.png")

    # 50% 확률로 배경 변경 (원본도 학습에 포함)
    if random.random() < 0.5:
        texture = random.choice(TEXTURES)
        frame = composite_bg(frame, mask, f"{TEXTURE_DIR}/{texture}")

    return frame
```

### 3. `replay_dr.py`에서 background DR 제거

배경 DR 오프라인 렌더링이 불필요해지므로 `--dr-type background` 옵션을 더 이상 사용하지 않는다.

---

## 기대 효과

| 항목 | 기존 | 개선 후 |
|------|------|---------|
| 저장 파일 수 (ep 1개, T=100) | 8,100장 | 200장 |
| 배경 DR 렌더링 시간 | ~80× | 0 (제거) |
| 텍스처 다양성 | 80가지 (고정) | 무한대 (학습마다 랜덤) |
| 학습 시 오버헤드 | 없음 | cv2.resize + 픽셀 대체 (무시할 수준) |

---

## 테이블탑 텍스처 처리

바닥(green-screen)과 달리 테이블탑은 SAPIEN material API로 변경하기 때문에
**마스크 기반 합성이 아닌 렌더링이 필요**하다.

옵션 A: 테이블탑 DR도 별도 오프라인 렌더링으로 유지 (현행 유지)
옵션 B: 테이블탑 마스크도 별도로 저장해서 동일하게 온라인 합성

현재 코드에서 `config["tabletop"]`은 `change_actor_texture()`로 처리되는데,
텍스처 이미지를 렌더러 없이 직접 마스크에 붙이면 조명 반응이 다를 수 있다.
우선 **바닥(floor)만 온라인 합성**으로 전환하고 테이블탑은 유지하는 것을 권장한다.

---

## 마이그레이션 순서

1. `replay_anchor.py`에 mask 저장 기능 추가
2. 기존 에피소드들에 대해 mask 재생성 (`replay_anchor.py` 재실행)
3. DataLoader에 `composite_bg()` 통합
4. `replay_dr.py`에서 `--dr-type background` 호출 제거
5. 기존 `bg_*/` 디렉토리 삭제
