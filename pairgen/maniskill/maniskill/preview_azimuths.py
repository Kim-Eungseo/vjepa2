"""Preview all 24 azimuth angles (0°~345°, 15° intervals) from the anchor front view.

Renders one frame per azimuth at default elevation and distance, saving
individual images to a preview directory. Use this to decide which azimuths
to exclude from the camera DR config.

Usage:
    python preview_azimuths.py \
        --traj-path /data1/maniskill/demos/LiftPegUpright-v1/motionplanning/joint_pos.h5 \
        --output-dir /home/sylee/maniskill/azimuth_preview
"""
import os
import json
import argparse
import numpy as np
import gymnasium as gym
import imageio
import mani_skill.envs
from mani_skill.utils import io_utils
from mani_skill.utils.sapien_utils import look_at
from transforms3d.quaternions import quat2mat


def get_default_camera_info(env):
    """Extract default camera eye/target from environment."""
    cam_configs = env.unwrapped._default_human_render_camera_configs
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

    # Estimate target by projecting look direction to z=0 plane area
    denom = look_dir[0] ** 2 + look_dir[1] ** 2
    if denom > 1e-6:
        t = -(look_dir[0] * p[0] + look_dir[1] * p[1]) / denom
    else:
        t = 1.0
    target = p + t * look_dir
    return p, target


def compute_anchor_from_default(eye, target):
    """Compute anchor (front view) camera parameters from default camera."""
    vec = np.array(eye) - np.array(target)
    radius = np.linalg.norm(vec)
    azimuth = np.arctan2(vec[1], vec[0])
    elevation = np.arcsin(np.clip(vec[2] / radius, -1.0, 1.0))
    return {
        "azimuth_rad": azimuth,
        "azimuth_deg": np.degrees(azimuth),
        "elevation_rad": elevation,
        "elevation_deg": np.degrees(elevation),
        "radius": radius,
        "target": target.tolist(),
    }


def get_camera_pose(target, azimuth_deg, elevation_rad, radius):
    """Compute camera Pose from spherical coordinates relative to target."""
    az_rad = np.radians(azimuth_deg)
    dx = radius * np.cos(elevation_rad) * np.cos(az_rad)
    dy = radius * np.cos(elevation_rad) * np.sin(az_rad)
    dz = radius * np.sin(elevation_rad)
    eye = np.array(target) + np.array([dx, dy, dz])
    return look_at(eye, target)


def main():
    parser = argparse.ArgumentParser(description="Preview all azimuth angles for camera DR")
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="azimuth_preview")
    parser.add_argument("--episode-idx", type=int, default=0)
    args = parser.parse_args()

    # Load trajectory metadata
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    task_id = json_data["env_info"]["env_id"]

    # Get default camera info
    tmp_env = gym.make(task_id, render_mode="rgb_array")
    tmp_env.reset()
    default_eye, default_target = get_default_camera_info(tmp_env)
    tmp_env.close()

    anchor = compute_anchor_from_default(default_eye, default_target)
    print(f"Default camera: eye={default_eye}, azimuth={anchor['azimuth_deg']:.1f}°")
    print(f"Anchor target: {anchor['target']}")
    print(f"Anchor elevation: {anchor['elevation_deg']:.1f}°, radius: {anchor['radius']:.3f}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load episode data for env state replay
    import h5py
    from mani_skill.trajectory.utils import dict_to_list_of_dicts

    h5_file = h5py.File(args.traj_path, "r")
    episode = json_data["episodes"][args.episode_idx]
    traj_id = f"traj_{episode['episode_id']}"

    reset_kwargs = episode["reset_kwargs"].copy()
    if "seed" in reset_kwargs and isinstance(reset_kwargs["seed"], list):
        reset_kwargs["seed"] = reset_kwargs["seed"][0]

    # Render each azimuth
    azimuths = list(range(0, 360, 15))  # 0, 15, 30, ..., 345

    for az in azimuths:
        pose = get_camera_pose(
            anchor["target"], az, anchor["elevation_rad"], anchor["radius"]
        )

        env_kwargs = json_data["env_info"]["env_kwargs"].copy()
        env_kwargs["render_mode"] = "rgb_array"
        env_kwargs["sim_backend"] = "physx_cpu"
        env_kwargs["num_envs"] = 1
        env_kwargs["human_render_camera_configs"] = {"render_camera": {"pose": pose}}

        env = gym.make(task_id, **env_kwargs)
        env.reset(**reset_kwargs)

        # Set to a mid-trajectory state so the robot is doing something interesting
        ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
        mid_step = len(ori_env_states) // 2
        env.unwrapped.set_state_dict(ori_env_states[mid_step])

        frame = env.render().cpu().numpy()[0]
        out_path = os.path.join(args.output_dir, f"az_{az:03d}.png")
        imageio.imsave(out_path, frame)
        print(f"  az={az:3d}° → {out_path}  ({frame.shape[1]}x{frame.shape[0]})")
        env.close()

    h5_file.close()

    print(f"\nDone! {len(azimuths)} images saved to {args.output_dir}/")
    print("각 이미지를 확인하고, 로봇 task가 잘 안 보이는 방위각을 제외하세요.")


if __name__ == "__main__":
    main()
