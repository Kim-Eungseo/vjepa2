"""Replay base trajectory with anchor (front-view) camera and save RGB frames.

For each episode:
  1. Auto-detects or loads anchor camera pose from configs.json
  2. Replays with --use-env-states (exact state reproduction)
  3. Saves per-frame PNG images to ep_N/anchor/

Usage:
    python replay_anchor.py \
        --traj-path /data1/maniskill/demos/LiftPegUpright-v1/motionplanning/joint_pos.h5 \
        --output-dir /data1/maniskill/datasets/LiftPegUpright-v1 \
        --count 2
"""
import os
import json
import argparse
import numpy as np
import h5py
import gymnasium as gym
import imageio
import mani_skill.envs
from mani_skill.utils import io_utils
from mani_skill.utils.sapien_utils import look_at
from mani_skill.trajectory.utils import dict_to_list_of_dicts
from tqdm import tqdm


def load_or_compute_anchor(configs_path, task_id):
    """Load anchor config from configs.json, or raise if not found."""
    if not os.path.exists(configs_path):
        raise FileNotFoundError(
            f"configs.json not found at {configs_path}.\n"
            f"Run generate_configs.py first to create it."
        )
    with open(configs_path, "r") as f:
        configs = json.load(f)
    return configs["anchor"]


def get_anchor_pose(anchor_info):
    """Build SAPIEN Pose from anchor camera config."""
    eye = np.array(anchor_info["eye"])
    target = np.array(anchor_info["target"])
    return look_at(eye, target)


def replay_episode(env, h5_file, episode, anchor_info, output_dir):
    """Replay a single episode and save anchor RGB frames."""
    episode_id = episode["episode_id"]
    traj_id = f"traj_{episode_id}"
    anchor_dir = os.path.join(output_dir, f"ep{episode_id:03d}", "anchor")
    obs_dir = os.path.join(anchor_dir, "obs")
    os.makedirs(obs_dir, exist_ok=True)

    # Reset and restore initial state
    reset_kwargs = episode["reset_kwargs"].copy()
    if "seed" in reset_kwargs and isinstance(reset_kwargs["seed"], list):
        reset_kwargs["seed"] = reset_kwargs["seed"][0]

    env.reset(**reset_kwargs)

    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    env.unwrapped.set_state_dict(ori_env_states[0])

    ori_actions = h5_file[traj_id]["actions"][:]

    # Save meta.json
    meta = {
        "type": "anchor",
        "episode_id": episode_id,
        "num_frames": len(ori_actions) + 1,
        "camera": anchor_info,
    }
    with open(os.path.join(anchor_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Render initial frame
    frame = env.render().cpu().numpy()[0]
    imageio.imsave(os.path.join(obs_dir, "frame_000.png"), frame)

    # Replay with env states
    for i, a in enumerate(ori_actions):
        env.step(a)
        env.unwrapped.set_state_dict(ori_env_states[i + 1])
        frame = env.render().cpu().numpy()[0]
        imageio.imsave(os.path.join(obs_dir, f"frame_{i+1:03d}.png"), frame)

    return len(ori_actions) + 1

def replay_worker(args_tuple):
    """Worker: replay a batch of episodes."""
    (traj_path, output_dir, task_id, anchor_info, anchor_pose,
     episode_indices, json_data, resolution) = args_tuple

    from multiprocessing import current_process
    pid = current_process().name

    env_kwargs = json_data["env_info"]["env_kwargs"].copy()
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = 1
    env_kwargs["human_render_camera_configs"] = {
        "shader_pack": "default",
        "render_camera": {"pose": anchor_pose, "width": resolution, "height": resolution},
    }

    env = gym.make(task_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)

    h5_file = h5py.File(traj_path, "r")
    episodes = json_data["episodes"]

    for idx in episode_indices:
        episode = episodes[idx]
        n_frames = replay_episode(env, h5_file, episode, anchor_info, output_dir)
        # print removed for tqdm

    env.close()
    h5_file.close()


def main():
    parser = argparse.ArgumentParser(description="Replay trajectory with anchor camera and save RGB frames")
    parser.add_argument("--traj-path", type=str, required=True,
                        help="Path to base trajectory h5 file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output directory (must contain configs.json)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of episodes to replay (default: all)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting episode index (default: 0)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolution for rendering (default: 512)")
    args = parser.parse_args()

    # Load configs
    configs_path = os.path.join(args.output_dir, "configs.json")
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    task_id = json_data["env_info"]["env_id"]

    anchor_info = load_or_compute_anchor(configs_path, task_id)
    anchor_pose = get_anchor_pose(anchor_info)

    print(f"Task: {task_id}")
    print(f"Anchor camera: eye={[f'{x:.3f}' for x in anchor_info['eye']]}, "
          f"elevation={anchor_info['elevation_deg']:.1f}°")
    print(f"Workers: {args.workers}")
    print()

    episodes = json_data["episodes"]
    start = args.start_idx
    end = len(episodes) if args.count is None else min(start + args.count, len(episodes))
    all_indices = list(range(start, end))

    print(f"Replaying episodes {start} to {end-1} ({len(all_indices)} episodes)")
    print(f"Output: {args.output_dir}")
    print()

    if args.workers <= 1:
        replay_worker((
            args.traj_path, args.output_dir, task_id,
            anchor_info, anchor_pose, all_indices, json_data, args.resolution
        ))
    else:
        from multiprocessing import Pool
        # Split episodes across workers: 1 job = 1 episode to make tqdm progress smooth
        jobs = []
        for idx in all_indices:
            jobs.append((
                args.traj_path, args.output_dir, task_id,
                anchor_info, anchor_pose, [idx], json_data, args.resolution
            ))

        with Pool(args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(replay_worker, jobs), total=len(jobs), desc="Anchor Replay Progress"):
                pass

    print(f"\nDone! Anchor RGB frames saved to {args.output_dir}/epNNN/anchor/")


if __name__ == "__main__":
    main()

