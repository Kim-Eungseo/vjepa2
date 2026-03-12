"""Replay trajectory with domain randomization — GPU-batched version.

For camera / lighting DR, batches multiple configs per episode into
parallel sub-envs so that rendering happens in a single GPU pass.
Background DR falls back to the sequential CPU approach.

Same CLI interface as replay_dr.py, with added --batch-size flag.

Usage:
    # Camera DR — batch 8 configs per GPU pass
    python replay_dr_fast.py \
        --traj-path /data1/maniskill/demos/LiftPegUpright-v1/motionplanning/joint_pos.h5 \
        --output-dir /data1/maniskill/datasets/LiftPegUpright-v1 \
        --dr-type camera --n-samples 50 --count 5 --batch-size 8

    # Lighting DR — batch all 8 configs at once
    python replay_dr_fast.py \
        --traj-path ... --output-dir ... \
        --dr-type lighting --count 5 --batch-size 8

    # Background DR — same as replay_dr.py (no GPU batching)
    python replay_dr_fast.py \
        --traj-path ... --output-dir ... \
        --dr-type background --n-samples 30 --count 5 --workers 4
"""
import os
import json
import argparse
import numpy as np
import h5py
import gymnasium as gym
import imageio
import torch
import cv2
import sapien
from tqdm import tqdm
import mani_skill.envs
import sapien.render
from mani_skill.utils import io_utils, sapien_utils
from mani_skill.trajectory.utils import dict_to_list_of_dicts
from multiprocessing import Pool, current_process


# ─── Episode directory helper ───────────────────────────────────────

def episode_dir_name(episode_id):
    return f"ep{int(episode_id):03d}"


# ─── Camera helpers ──────────────────────────────────────────────────

def get_camera_pose(anchor_info, az_delta, el_delta, dist_scale):
    """Compute camera Pose from anchor + deltas."""
    target = np.array(anchor_info["target"])
    base_el = np.radians(anchor_info["elevation_deg"])
    base_radius = anchor_info["radius"]

    new_az = np.radians(az_delta)
    new_el = base_el + np.radians(el_delta)
    new_el = np.clip(new_el, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    new_radius = base_radius * dist_scale

    dx = new_radius * np.cos(new_el) * np.cos(new_az)
    dy = new_radius * np.cos(new_el) * np.sin(new_az)
    dz = new_radius * np.sin(new_el)
    eye = target + np.array([dx, dy, dz])
    return sapien_utils.look_at(eye, target)


# ─── Lighting helpers ────────────────────────────────────────────────

def apply_lighting_to_subscene(sub_scene, light_config):
    """Apply lighting config to a single sub-scene."""
    ambient = np.array(light_config["ambient_light"])
    dir_color = np.array(light_config["directional_light_color"])
    sub_scene.render_system.ambient_light = ambient
    for entity in sub_scene.entities:
        for comp in entity.components:
            if "Light" in type(comp).__name__:
                comp.color = dir_color


# ─── Background helpers (unchanged from replay_dr.py) ───────────────

def change_actor_texture(actor, texture_file):
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


def get_mask_from_obs(obs, seg_ids):
    cam_name = list(obs["sensor_data"].keys())[0]
    seg = obs["sensor_data"][cam_name]["segmentation"]
    mask = torch.isin(seg[0, :, :, 0], seg_ids.to(seg.device))
    return mask.cpu().numpy().astype(np.uint8)


# ─── Per-episode sampling (identical to replay_dr.py) ────────────────

def sample_config_ids(dr_type, all_configs, n_samples, episode_id, base_seed=42):
    """Sample config IDs for a single episode."""
    rng = np.random.default_rng(base_seed + episode_id)

    if dr_type == "camera":
        all_ids = sorted(all_configs["camera"].keys(), key=int)
        weights = np.array([all_configs["camera_sampling_weights"][k] for k in all_ids])
        weights /= weights.sum()
        n = min(n_samples, len(all_ids))
        sampled = rng.choice(all_ids, size=n, replace=False, p=weights)
    elif dr_type == "background":
        all_ids = sorted(all_configs["background"].keys(), key=int)
        n = min(n_samples, len(all_ids))
        sampled = rng.choice(all_ids, size=n, replace=False)
    else:  # lighting: use all
        sampled = sorted(all_configs["lighting"].keys(), key=int)

    return sorted(sampled, key=int)


# ─── State dict broadcast ───────────────────────────────────────────

def broadcast_state_dict(single_state, num_envs):
    """Replicate a single-env state dict to num_envs copies (batched).

    Each ndarray of shape (D,) or (1, D) is tiled to (num_envs, D).
    """
    out = {}
    for key, val in single_state.items():
        if isinstance(val, dict):
            out[key] = broadcast_state_dict(val, num_envs)
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                out[key] = np.tile(val[None, :], (num_envs, 1))
            elif val.ndim >= 2 and val.shape[0] == 1:
                out[key] = np.tile(val, (num_envs,) + (1,) * (val.ndim - 1))
            else:
                out[key] = np.tile(val, (num_envs, 1))
        else:
            out[key] = val
    return out


# ═══════════════════════════════════════════════════════════════════
#  GPU-BATCHED replay for camera / lighting DR
# ═══════════════════════════════════════════════════════════════════

def _gpu_batch_replay_episode(
    env, h5_file, episode, dr_type, config_batch, batch_size,
    anchor_info, output_dir
):
    """Replay one episode with a batch of DR configs using GPU sub-envs.

    All sub-envs share the same episode states, but each has a different
    camera pose or lighting config. env.render() renders all at once.

    Args:
        env: gym env with num_envs = batch_size
        h5_file: opened HDF5 trajectory file
        episode: episode metadata dict
        dr_type: "camera" or "lighting"
        config_batch: list of (config_id, config_dict), len <= batch_size
        batch_size: env.num_envs (may be > len(config_batch) for last batch)
        anchor_info: anchor camera info from configs.json
        output_dir: base output directory
    """
    episode_id = episode["episode_id"]
    traj_id = f"traj_{episode_id}"
    dir_prefix = {"camera": "cam", "lighting": "lighting"}[dr_type]
    actual_count = len(config_batch)  # may be < batch_size for last batch

    # ── Load trajectory data ──
    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    ori_actions = h5_file[traj_id]["actions"][:]

    # ── Reset env ──
    reset_kwargs = episode["reset_kwargs"].copy()
    if "seed" in reset_kwargs and isinstance(reset_kwargs["seed"], list):
        reset_kwargs["seed"] = reset_kwargs["seed"][0]
    env.reset(**reset_kwargs)

    # ── Set initial state (broadcast to all sub-envs) ──
    batched_state = broadcast_state_dict(ori_env_states[0], batch_size)
    env.unwrapped.set_state_dict(batched_state)

    # ── Apply per-sub-env DR configs ──
    scene = env.unwrapped.scene
    # Camera sensor (.camera → RenderCamera wrapping per-sub-scene components)
    render_camera = env.unwrapped._human_render_cameras["render_camera"].camera

    if dr_type == "camera":
        for idx in range(batch_size):
            src_idx = min(idx, actual_count - 1)  # pad with last config
            _, config = config_batch[src_idx]
            pose = get_camera_pose(
                anchor_info,
                config["azimuth_delta"],
                config["elevation_delta"],
                config["distance_scale"],
            )
            render_camera._render_cameras[idx].set_local_pose(pose.sp)
    elif dr_type == "lighting":
        for idx in range(batch_size):
            src_idx = min(idx, actual_count - 1)
            _, config = config_batch[src_idx]
            apply_lighting_to_subscene(scene.sub_scenes[idx], config)

    # ── Create output dirs + save meta.json (only for real configs) ──
    dr_dirs = []
    for idx in range(actual_count):
        cid, config = config_batch[idx]
        dr_dir = os.path.join(
            output_dir, episode_dir_name(episode_id),
            f"{dir_prefix}_{int(cid):03d}"
        )
        obs_dir = os.path.join(dr_dir, "obs")
        os.makedirs(obs_dir, exist_ok=True)
        dr_dirs.append(dr_dir)

        meta = {
            "type": dr_type,
            "config_id": int(cid),
            "episode_id": episode_id,
            "num_frames": len(ori_actions) + 1,
            "config": config,
        }
        with open(os.path.join(dr_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Render initial frame ──
    frames = env.render().cpu().numpy()  # (batch_size, H, W, 3)
    for idx in range(actual_count):
        imageio.imsave(os.path.join(dr_dirs[idx], "obs", "frame_000.png"), frames[idx])

    # ── Replay steps ──
    for step_i, a in enumerate(ori_actions):
        # Broadcast action to all sub-envs
        action_batch = np.tile(a, (batch_size, 1))
        env.step(action_batch)

        # Force all sub-envs to exact state
        batched_state = broadcast_state_dict(ori_env_states[step_i + 1], batch_size)
        env.unwrapped.set_state_dict(batched_state)

        # Render all sub-envs at once
        frames = env.render().cpu().numpy()
        for idx in range(actual_count):
            imageio.imsave(
                os.path.join(dr_dirs[idx], "obs", f"frame_{step_i+1:03d}.png"),
                frames[idx],
            )

    return len(ori_actions) + 1


def gpu_batch_replay_all(
    traj_path, output_dir, dr_type, all_configs, anchor_info,
    json_data, episode_indices, n_samples, base_seed, explicit_config_ids,
    batch_size,
):
    """Main GPU-batched replay loop for camera / lighting DR."""

    task_id = json_data["env_info"]["env_id"]
    env_kwargs = json_data["env_info"]["env_kwargs"].copy()
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["num_envs"] = batch_size

    # Default camera pose (will be overridden per sub-env)
    anchor_pose = sapien_utils.look_at(
        np.array(anchor_info["eye"]), np.array(anchor_info["target"])
    )
    env_kwargs["human_render_camera_configs"] = {
        "shader_pack": "default",
        "render_camera": {"pose": anchor_pose},
    }

    print(f"Creating GPU-batched env with num_envs={batch_size}...")
    env = gym.make(task_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)

    h5_file = h5py.File(traj_path, "r")
    episodes = json_data["episodes"]

    for ep_idx in episode_indices:
        episode = episodes[ep_idx]
        episode_id = episode["episode_id"]

        # Determine config IDs (identical logic to replay_dr.py)
        if explicit_config_ids is not None:
            config_ids = explicit_config_ids
        else:
            config_ids = sample_config_ids(
                dr_type, all_configs, n_samples, episode_id, base_seed
            )

        config_items = [(cid, all_configs[dr_type][cid]) for cid in config_ids]

        # Process in batches
        n_batches = (len(config_items) + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches), desc=f"{episode_dir_name(episode_id)} Batches", leave=False):
            config_batch = config_items[bi * batch_size : (bi + 1) * batch_size]

            n_frames = _gpu_batch_replay_episode(
                env, h5_file, episode, dr_type, config_batch, batch_size,
                anchor_info, output_dir,
            )

            batch_cids = [cid for cid, _ in config_batch]
            # Print statement removed to avoid cluttering tqdm output

    env.close()
    h5_file.close()


# ═══════════════════════════════════════════════════════════════════
#  CPU sequential replay for background DR (same as replay_dr.py)
# ═══════════════════════════════════════════════════════════════════

def _setup_env_bg(task_id, env_kwargs_base, config, anchor_info):
    """Create environment for background DR."""
    env_kwargs = env_kwargs_base.copy()
    need_gs = config.get("floor") is not None
    env_kwargs["obs_mode"] = "rgb+segmentation"
    anchor_pose = sapien_utils.look_at(
        np.array(anchor_info["eye"]), np.array(anchor_info["target"])
    )
    env_kwargs["human_render_camera_configs"] = {
        "shader_pack": "default",
        "render_camera": {"pose": anchor_pose},
    }
    env_kwargs["sensor_configs"] = {
        "base_camera": {
            "pose": anchor_pose, "width": 512, "height": 512, "fov": 1,
        }
    }

    env = gym.make(task_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)
    return env, need_gs


def _replay_episode_bg(env, h5_file, episode, config_id, config,
                       anchor_info, texture_dir, output_dir, need_gs, pid):
    """Replay a single episode with background DR config."""
    episode_id = episode["episode_id"]
    traj_id = f"traj_{episode_id}"

    dr_dir = os.path.join(
        output_dir, episode_dir_name(episode_id),
        f"bg_{int(config_id):03d}"
    )
    obs_dir = os.path.join(dr_dir, "obs")
    os.makedirs(obs_dir, exist_ok=True)

    reset_kwargs = episode["reset_kwargs"].copy()
    if "seed" in reset_kwargs and isinstance(reset_kwargs["seed"], list):
        reset_kwargs["seed"] = reset_kwargs["seed"][0]

    obs, _ = env.reset(**reset_kwargs)
    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    env.unwrapped.set_state_dict(ori_env_states[0])

    floor_seg_ids = None
    floor_bg_image = None
    scene = env.unwrapped.scene
    if config.get("tabletop") is not None:
        tex_path = os.path.join(texture_dir, config["tabletop"])
        change_actor_texture(scene.actors["table-workspace"], tex_path)
    if config.get("floor") is not None:
        ground_actor = scene.actors["ground"]
        floor_seg_ids = ground_actor.per_scene_id.to(dtype=torch.int16)
        img = cv2.imread(os.path.join(texture_dir, config["floor"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        floor_bg_image = cv2.resize(img, (512, 512))

    ori_actions = h5_file[traj_id]["actions"][:]

    meta = {
        "type": "background",
        "config_id": int(config_id),
        "episode_id": episode_id,
        "num_frames": len(ori_actions) + 1,
        "config": config,
    }
    with open(os.path.join(dr_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def render_frame(obs=None):
        frame = env.render().cpu().numpy()[0]
        if need_gs and obs is not None and floor_seg_ids is not None:
            mask = get_mask_from_obs(obs, floor_seg_ids)
            result = frame.copy()
            result[mask == 1] = floor_bg_image[mask == 1]
            frame = result
        return frame

    obs = env.unwrapped.get_obs()
    frame = render_frame(obs)
    imageio.imsave(os.path.join(obs_dir, "frame_000.png"), frame)

    for i, a in enumerate(ori_actions):
        obs, _, _, _, _ = env.step(a)
        env.unwrapped.set_state_dict(ori_env_states[i + 1])
        obs = env.unwrapped.get_obs()
        frame = render_frame(obs)
        imageio.imsave(os.path.join(obs_dir, f"frame_{i+1:03d}.png"), frame)


def bg_replay_worker(args_tuple):
    """Process one episode for background DR (CPU sequential)."""
    (traj_path, output_dir, ep_idx,
     all_configs, anchor_info, texture_dir,
     json_data, n_samples, base_seed, explicit_config_ids) = args_tuple

    task_id = json_data["env_info"]["env_id"]
    env_kwargs_base = json_data["env_info"]["env_kwargs"].copy()
    env_kwargs_base["render_mode"] = "rgb_array"
    env_kwargs_base["sim_backend"] = "physx_cpu"
    env_kwargs_base["num_envs"] = 1

    episode = json_data["episodes"][ep_idx]
    episode_id = episode["episode_id"]

    if explicit_config_ids is not None:
        config_ids = explicit_config_ids
    else:
        config_ids = sample_config_ids(
            "background", all_configs, n_samples, episode_id, base_seed)

    h5_file = h5py.File(traj_path, "r")
    pid = current_process().name

    for cid in config_ids:
        config = all_configs["background"][cid]
        env, need_gs = _setup_env_bg(
            task_id, env_kwargs_base, config, anchor_info)
        _replay_episode_bg(
            env, h5_file, episode, cid, config,
            anchor_info, texture_dir, output_dir, need_gs, pid)
        env.close()

    h5_file.close()
    return episode_id, len(config_ids)


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay trajectory with domain randomization (GPU-batched)"
    )
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset dir (must contain configs.json)")
    parser.add_argument("--dr-type", type=str, required=True,
                        choices=["camera", "lighting", "background"])
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of configs to sample per episode "
                             "(camera default=50, bg default=30, lighting=all)")
    parser.add_argument("--config-ids", type=str, default=None,
                        help="Explicit config IDs (comma-separated), overrides sampling")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of episodes (default: all)")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for per-episode sampling")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of configs to process in parallel via GPU "
                             "(camera/lighting only). Default: 8")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (background DR only)")
    parser.add_argument("--texture-dir", type=str,
                        default="/home/sylee/maniskill/texture-images",
                        help="Directory with texture images (for background DR)")
    args = parser.parse_args()

    # Load configs
    configs_path = os.path.join(args.output_dir, "configs.json")
    with open(configs_path) as f:
        all_configs = json.load(f)

    anchor_info = all_configs["anchor"]

    # Load trajectory metadata
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    episodes = json_data["episodes"]

    start = args.start_idx
    end = len(episodes) if args.count is None else min(start + args.count, len(episodes))
    episode_indices = list(range(start, end))

    # Determine samples per episode
    defaults = {"camera": 50, "lighting": 8, "background": 30}
    n_samples = args.n_samples or defaults[args.dr_type]

    # Explicit config IDs override sampling
    explicit_ids = None
    if args.config_ids:
        explicit_ids = [s.strip() for s in args.config_ids.split(",")]

    total_configs = len(all_configs[args.dr_type])
    print(f"Task: {json_data['env_info']['env_id']}")
    print(f"DR type: {args.dr_type} ({total_configs} total configs available)")
    if explicit_ids:
        print(f"Explicit config IDs: {explicit_ids}")
    else:
        print(f"Sampling: {n_samples} per episode (seed={args.seed})")
    print(f"Episodes: {start}~{end-1} ({len(episode_indices)} total)")

    if args.dr_type in ("camera", "lighting"):
        print(f"Mode: GPU-batched (batch_size={args.batch_size})")
        print()

        gpu_batch_replay_all(
            args.traj_path, args.output_dir, args.dr_type,
            all_configs, anchor_info, json_data,
            episode_indices, n_samples, args.seed, explicit_ids,
            args.batch_size,
        )
    else:
        # Background DR: CPU multiprocessing (same as replay_dr.py)
        print(f"Mode: CPU sequential (workers={args.workers})")
        print()

        jobs = []
        for ep_idx in episode_indices:
            jobs.append((
                args.traj_path, args.output_dir, ep_idx,
                all_configs, anchor_info, args.texture_dir,
                json_data, n_samples, args.seed, explicit_ids,
            ))

        if args.workers <= 1:
            for job in tqdm(jobs, desc="Background DR Progress"):
                bg_replay_worker(job)
        else:
            with Pool(args.workers) as pool:
                for _ in tqdm(pool.imap_unordered(bg_replay_worker, jobs), total=len(jobs), desc="Background DR Progress"):
                    pass

    print(f"\nDone! {args.dr_type} DR completed for {len(episode_indices)} episodes")


if __name__ == "__main__":
    main()
