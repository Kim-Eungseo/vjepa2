"""Replay trajectory with background texture domain randomization.

Applies texture changes to tabletop, floor, and wall based on a config ID
from bg_configs.json (729 = 9x9x9 combinations).

- Tabletop (table-workspace): SAPIEN material API (set_base_color_texture)
- Floor (ground actor): segmentation mask based green screening
- Wall (background, seg ID 0): segmentation mask based green screening
- null = keep original appearance

RGB and segmentation are both taken from the same sensor camera, so the
mask always aligns with the frame regardless of camera pose/fov/resolution.
"""
import os
import json
import argparse
import h5py
import numpy as np
import gymnasium as gym
import imageio
import torch
import cv2
import sapien.render
import mani_skill.envs
from mani_skill.utils import io_utils, sapien_utils
from mani_skill.trajectory.utils import dict_to_list_of_dicts


# ─── Actor texture helpers ───────────────────────────────────────────

def change_actor_texture(actor, texture_file):
    """Apply an image texture to all visual parts of an actor (SAPIEN material API).

    Works for actors loaded via builder pipeline (e.g. table-workspace from .glb).
    """
    texture = sapien.render.RenderTexture2D(filename=texture_file, mipmap_levels=4)
    for obj in actor._objs:
        for comp in obj.components:
            if isinstance(comp, sapien.render.RenderBodyComponent):
                for shape in comp.render_shapes:
                    for part in shape.parts:
                        part.material.set_base_color_texture(texture)

                        part.material.set_normal_texture(None)
                        part.material.set_metallic_texture(None)
                        part.material.set_roughness_texture(None)
                        part.material.set_emission_texture(None)

                        part.material.set_base_color([1.0, 1.0, 1.0, 1.0])
                        part.material.set_roughness(0.5)


# ─── Green-screen helpers ────────────────────────────────────────────

def get_rgb_from_obs(obs):
    """Extract RGB frame from sensor observation.

    Returns:
        numpy array [H, W, 3], uint8
    """
    cam_name = list(obs["sensor_data"].keys())[0]
    rgb = obs["sensor_data"][cam_name]["rgb"]  # [B, H, W, 3]
    return rgb[0].cpu().numpy()


def get_mask_from_obs(obs, seg_ids):
    """Extract binary mask for the given segmentation IDs from observation.

    Returns:
        numpy mask [H, W], 1 = matched, 0 = not matched
    """
    cam_name = list(obs["sensor_data"].keys())[0]
    seg = obs["sensor_data"][cam_name]["segmentation"]  # [B, H, W, 1]
    mask = torch.isin(seg[0, :, :, 0], seg_ids.to(seg.device))  # [H, W]
    return mask.cpu().numpy().astype(np.uint8)


def apply_green_screen_image(frame, mask, bg_image):
    """Replace masked pixels in a frame with an image."""
    result = frame.copy()
    result[mask == 1] = bg_image[mask == 1]
    return result


# ─── Core replay function ────────────────────────────────────────────

def replay_with_bg(config_id, config, texture_dir, h5_path, episode_idx=0):
    """Replay a trajectory with the specified background textures applied.

    Args:
        config_id: Background config ID (for logging/output naming).
        config: Dict with 'tabletop', 'floor', 'wall' (filename or null).
        texture_dir: Base directory for texture image files.
        h5_path: Path to trajectory HDF5 file.
        episode_idx: Episode index to replay (default: 0).
    """
    json_path = h5_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    env_id = json_data["env_info"]["env_id"]

    env_kwargs = json_data["env_info"]["env_kwargs"].copy()
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = 1

    # Determine if green screening is needed (floor or wall texture change)
    need_gs = config["floor"] is not None or config["wall"] is not None

    # Always use rgb+segmentation so we can extract RGB from the sensor camera
    # (ensures mask and frame always come from the same camera)
    env_kwargs["obs_mode"] = "rgb+segmentation"

    # Camera setup (use same posed camera for both rendering and segmentation)
    human_cam_pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
    env_kwargs["human_render_camera_configs"] = {
        "pose": human_cam_pose, "width": 512, "height": 512, "fov": 1,
    }
    env_kwargs["sensor_configs"] = {
        "base_camera": {
            "pose": human_cam_pose, "width": 512, "height": 512, "fov": 1,
        }
    }

    env = gym.make(env_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)

    # --- Load trajectory data ---
    h5_file = h5py.File(h5_path, "r")
    episodes = json_data["episodes"]
    episode = episodes[episode_idx]
    traj_id = f"traj_{episode['episode_id']}"

    reset_kwargs = episode["reset_kwargs"].copy()
    if "seed" in reset_kwargs and isinstance(reset_kwargs["seed"], list):
        reset_kwargs["seed"] = reset_kwargs["seed"][0]

    print(f"[BG {config_id}] Task: {env_id} | {config['description']}")
    obs, _ = env.reset(**reset_kwargs)

    # Restore the initial environment state from the trajectory
    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    env.unwrapped.set_state_dict(ori_env_states[0])

    # --- Apply tabletop texture (SAPIEN material API) ---
    scene = env.unwrapped.scene
    if config["tabletop"] is not None:
        table_actor = scene.actors["table-workspace"]
        tex_path = os.path.join(texture_dir, config["tabletop"])
        change_actor_texture(table_actor, tex_path)
        print(f"  Tabletop: {config['tabletop']}")
    else:
        print("  Tabletop: no_change")

    # --- Prepare green-screen resources ---
    wall_seg_ids = None
    floor_seg_ids = None
    wall_bg_image = None
    floor_bg_image = None

    if config["wall"] is not None:
        wall_seg_ids = torch.tensor([0], dtype=torch.int16)  # background = seg ID 0
        img = cv2.imread(os.path.join(texture_dir, config["wall"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wall_bg_image = cv2.resize(img, (512, 512))
        print(f"  Wall: {config['wall']}")
    else:
        print("  Wall: no_change")

    if config["floor"] is not None:
        ground_actor = scene.actors["ground"]
        floor_seg_ids = ground_actor.per_scene_id.to(dtype=torch.int16)
        img = cv2.imread(os.path.join(texture_dir, config["floor"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        floor_bg_image = cv2.resize(img, (512, 512))
        print(f"  Floor: {config['floor']}")
    else:
        print("  Floor: no_change")

    # --- Green-screen apply function ---
    def apply_gs(frame, obs):
        """Apply green-screening for wall and/or floor (only if configured)."""
        if wall_seg_ids is not None:
            wall_mask = get_mask_from_obs(obs, wall_seg_ids)
            frame = apply_green_screen_image(frame, wall_mask, wall_bg_image)
        if floor_seg_ids is not None:
            floor_mask = get_mask_from_obs(obs, floor_seg_ids)
            frame = apply_green_screen_image(frame, floor_mask, floor_bg_image)
        return frame

    # --- Replay and record ---
    ori_actions = h5_file[traj_id]["actions"][:]
    print(f"  Replaying {len(ori_actions)} actions...")

    # First frame
    obs = env.unwrapped.get_obs()
    frame = env.render().cpu().numpy()[0]
    if need_gs:
        frame = apply_gs(frame, obs)
    frames = [frame]

    for i, a in enumerate(ori_actions):
        obs, _, _, _, _ = env.step(a)
        env.unwrapped.set_state_dict(ori_env_states[i + 1])
        
        frame = env.render().cpu().numpy()[0]
        obs = env.unwrapped.get_obs()
        if need_gs:
            frame = apply_gs(frame, obs)

        frames.append(frame)
        if i % 10 == 0:
            print(f"  Stepping {i}")

    # --- Save video ---
    out_dir = os.path.join(os.path.dirname(h5_path), "bg_replay")
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"ep{episode['episode_id']}_bg{config_id}.mp4")
    imageio.mimsave(out_video, frames, fps=env.unwrapped.control_freq)
    env.close()
    h5_file.close()
    print(f"  Done! Saved to {out_video}")


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay trajectory with background texture domain randomization"
    )
    parser.add_argument(
        "--config-json", type=str, default="bg_configs.json",
        help="Path to bg_configs.json"
    )
    parser.add_argument(
        "--config-id", type=str, required=True,
        help="Background config ID (0-728) or 'all' to run all configs"
    )
    parser.add_argument(
        "--h5-path", type=str,
        default="/data1/maniskill/downloaded_demos/PickCube-v1/motionplanning/trajectory.h5",
        help="Path to trajectory HDF5 file"
    )
    parser.add_argument(
        "--episode-idx", type=int, default=0,
        help="Episode index to replay"
    )
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        bg_data = json.load(f)

    texture_dir = bg_data["texture_dir"]
    all_configs = bg_data["configs"]

    if args.config_id == "all":
        config_ids = sorted(all_configs.keys(), key=int)
    else:
        config_ids = [args.config_id]

    print(f"Total configs to run: {len(config_ids)}")

    for config_key in config_ids:
        if config_key not in all_configs:
            raise ValueError(
                f"Config ID {config_key} not found. Valid range: 0-{len(all_configs)-1}"
            )
        config = all_configs[config_key]
        replay_with_bg(int(config_key), config, texture_dir, args.h5_path, args.episode_idx)


if __name__ == "__main__":
    main()
