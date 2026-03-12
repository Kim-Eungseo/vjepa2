"""Generate unified configs.json with ALL possible config IDs.

Stores every valid combination for camera, lighting, and background
with fixed global IDs. Per-episode sampling happens at replay time.

- Camera: 129 combos (12 non-anchor azimuths × 5 elevations × 2 distances + 9 anchor-az variants)
- Lighting: 8 presets
- Background: 80 combos (9 tabletop × 9 floor - 1 original)

Usage:
    python generate_configs.py \
        --traj-path /data1/maniskill/demos/LiftPegUpright-v1/motionplanning/joint_pos.h5 \
        --output-dir /data1/maniskill/datasets/LiftPegUpright-v1
"""
import os
import json
import argparse
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import io_utils
from transforms3d.quaternions import quat2mat


# ─── Episode directory helper ───────────────────────────────────────

def episode_dir_name(episode_id):
    """Consistent episode directory naming: ep000, ep001, ..."""
    return f"ep{int(episode_id):03d}"


# ─── Camera Anchor Detection ────────────────────────────────────────

def get_default_camera_info(task_id):
    """Extract default camera eye/target from environment."""
    tmp_env = gym.make(task_id, render_mode="rgb_array")
    tmp_env.reset()

    cam_configs = tmp_env.unwrapped._default_human_render_camera_configs
    if isinstance(cam_configs, (list, tuple)):
        cfg = cam_configs[0]
    elif isinstance(cam_configs, dict):
        cfg = list(cam_configs.values())[0]
    else:
        cfg = cam_configs

    p = cfg.pose.p.cpu().numpy()[0]
    q = cfg.pose.q.cpu().numpy()[0]
    mat = quat2mat(q)
    look_dir = mat[:, 0]  # SAPIEN forward = +x

    denom = look_dir[0] ** 2 + look_dir[1] ** 2
    if denom > 1e-6:
        t = -(look_dir[0] * p[0] + look_dir[1] * p[1]) / denom
    else:
        t = 1.0
    target = p + t * look_dir
    tmp_env.close()
    return p, target


def compute_anchor(eye, target):
    """Compute anchor camera parameters (front view) from default camera."""
    vec = np.array(eye) - np.array(target)
    radius = float(np.linalg.norm(vec))
    azimuth = float(np.arctan2(vec[1], vec[0]))
    elevation = float(np.arcsin(np.clip(vec[2] / radius, -1.0, 1.0)))

    anchor_az = 0.0
    anchor_eye = np.array(target) + np.array([
        radius * np.cos(elevation) * np.cos(anchor_az),
        radius * np.cos(elevation) * np.sin(anchor_az),
        radius * np.sin(elevation),
    ])

    return {
        "eye": anchor_eye.tolist(),
        "target": [float(x) for x in target],
        "azimuth_deg": 0.0,
        "elevation_deg": float(np.degrees(elevation)),
        "radius": radius,
        "default_azimuth_deg": float(np.degrees(azimuth)),
        "azimuth_delta_from_default": float(np.degrees(-azimuth)),
    }


# ─── Camera Configs (ALL combos) ────────────────────────────────────

def generate_all_camera_configs():
    """Generate ALL valid camera configs with fixed global IDs.

    129 total = 12 non-anchor azimuths × 5 elevations × 2 distances
              + 9 anchor-azimuth variants (az=0, excluding anchor itself)
    """
    azimuths = [15, 30, 45, 60, 75, 90, 270, 285, 300, 315, 330, 345]
    elevations = [-15, 0, 15, 30, 45]
    distances = [1.0, 2.0]

    configs = {}
    idx = 0

    # Non-anchor azimuths (12 × 5 × 2 = 120)
    for az in azimuths:
        for el in elevations:
            for dist in distances:
                configs[str(idx)] = {
                    "azimuth_delta": int(az),
                    "elevation_delta": int(el),
                    "distance_scale": float(dist),
                }
                idx += 1

    # Anchor azimuth (0°) with non-anchor elevation/distance (9)
    for el in elevations:
        for dist in distances:
            if el == 0 and dist == 1.0:
                continue  # skip exact anchor
            configs[str(idx)] = {
                "azimuth_delta": 0,
                "elevation_delta": int(el),
                "distance_scale": float(dist),
            }
            idx += 1

    return configs


# ─── Lighting Configs ────────────────────────────────────────────────

def generate_all_lighting_configs():
    """Return 8 lighting presets with fixed IDs."""
    return {
        "0": {"name": "bright_warm", "ambient_light": [0.6, 0.55, 0.3], "directional_light_color": [1.2, 1.1, 0.6]},
        "1": {"name": "green_tint", "ambient_light": [0.15, 0.4, 0.15], "directional_light_color": [0.4, 1.0, 0.4]},
        "2": {"name": "pink_tint", "ambient_light": [0.5, 0.1, 0.25], "directional_light_color": [1.2, 0.3, 0.6]},
        "3": {"name": "blue_purple", "ambient_light": [0.1, 0.1, 0.45], "directional_light_color": [0.3, 0.3, 1.1]},
        "4": {"name": "cyan_teal", "ambient_light": [0.1, 0.4, 0.4], "directional_light_color": [0.3, 1.0, 0.95]},
        "5": {"name": "warm_sunset", "ambient_light": [0.55, 0.25, 0.08], "directional_light_color": [1.2, 0.5, 0.15]},
        "6": {"name": "very_dim", "ambient_light": [0.06, 0.06, 0.06], "directional_light_color": [0.2, 0.2, 0.2]},
        "7": {"name": "overexposed", "ambient_light": [0.8, 0.8, 0.8], "directional_light_color": [1.5, 1.5, 1.5]},
    }


# ─── Background Configs (ALL combos) ────────────────────────────────

TEXTURES = [
    "bricks.jpg", "cliffdesert.jpg", "cobblestone.png",
    "fabricclothes.jpg", "fabricpattern.png", "fabricsuedefine.jpg",
    "fabrictarpplastic.png", "metal.png",
]


def generate_all_background_configs():
    """Generate ALL (tabletop × floor) combos excluding (None, None).

    80 total = 9 × 9 - 1
    """
    tabletop_options = [None] + TEXTURES
    floor_options = [None] + TEXTURES

    configs = {}
    idx = 0
    for tab in tabletop_options:
        for floor in floor_options:
            if tab is None and floor is None:
                continue
            configs[str(idx)] = {
                "tabletop": tab,
                "floor": floor,
                "wall": None,
                "description": f"table={tab or 'original'}, floor={floor or 'original'}",
            }
            idx += 1

    return configs


# ─── Sampling weights (for replay_dr.py) ────────────────────────────

def compute_camera_sampling_weights(camera_configs):
    """Compute anchor-biased weights for all camera configs.

    Stored in configs.json so replay_dr.py can sample without recomputing.
    """
    weights = {}
    for cid, cfg in camera_configs.items():
        az = cfg["azimuth_delta"]
        el = cfg["elevation_delta"]
        dist = cfg["distance_scale"]

        az_diff = min(abs(az), 360 - abs(az))
        w_az = float(np.exp(-(az_diff ** 2) / (2 * 60 ** 2)))
        w_el = float(np.exp(-(el ** 2) / (2 * 25 ** 2)))
        w_dist = 0.7 if dist == 1.0 else 0.3
        weights[cid] = w_az * w_el * w_dist

    # Normalize
    total = sum(weights.values())
    for k in weights:
        weights[k] = round(weights[k] / total, 6)

    return weights


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate configs.json with all possible config IDs"
    )
    parser.add_argument("--traj-path", type=str, required=True,
                        help="Path to base trajectory h5 file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output dataset directory")
    args = parser.parse_args()

    # Load trajectory metadata
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    task_id = json_data["env_info"]["env_id"]

    print(f"Task: {task_id}")
    print(f"Detecting anchor camera...")

    # Auto-detect anchor
    default_eye, default_target = get_default_camera_info(task_id)
    anchor = compute_anchor(default_eye, default_target)

    print(f"  Default camera: azimuth={anchor['default_azimuth_deg']:.1f}°")
    print(f"  Anchor (front): eye={[f'{x:.3f}' for x in anchor['eye']]}, "
          f"elevation={anchor['elevation_deg']:.1f}°, radius={anchor['radius']:.3f}")
    print()

    # Generate all configs
    camera_configs = generate_all_camera_configs()
    lighting_configs = generate_all_lighting_configs()
    background_configs = generate_all_background_configs()
    camera_weights = compute_camera_sampling_weights(camera_configs)

    print(f"Camera configs: {len(camera_configs)} (all combos)")
    print(f"Lighting configs: {len(lighting_configs)}")
    print(f"Background configs: {len(background_configs)} (all combos)")

    # Build unified config
    unified_config = {
        "task_id": task_id,
        "base_trajectory": os.path.abspath(args.traj_path),
        "anchor": anchor,
        "camera": camera_configs,
        "camera_sampling_weights": camera_weights,
        "lighting": lighting_configs,
        "background": background_configs,
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "configs.json")
    with open(config_path, "w") as f:
        json.dump(unified_config, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {config_path}")


if __name__ == "__main__":
    main()
