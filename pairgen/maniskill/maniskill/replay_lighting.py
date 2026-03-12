import os
import json
import argparse
import h5py
import numpy as np
import gymnasium as gym
import imageio
import mani_skill.envs
from mani_skill.utils import io_utils
from mani_skill.trajectory.utils import dict_to_list_of_dicts


def apply_lighting(env, ambient_light, directional_light_color):
    """Apply lighting color to all sub-scenes in the environment.

    Args:
        env: ManiSkill environment (unwrapped access used internally).
        ambient_light: [R, G, B] values for ambient light.
        directional_light_color: [R, G, B] values for directional lights.
    """
    scene = env.unwrapped.scene
    for sub_scene in scene.sub_scenes:
        # Set ambient light color
        sub_scene.render_system.ambient_light = np.array(ambient_light)

        # Set directional light colors
        for entity in sub_scene.entities:
            for comp in entity.components:
                comp_type = type(comp).__name__
                if "Light" in comp_type:
                    comp.color = np.array(directional_light_color)


def replay_with_lighting(config_id, config, h5_path, episode_idx=0):
    """Replay a trajectory with the specified lighting color applied.

    Args:
        config_id: Lighting config ID (for logging/output naming).
        config: Dict with 'name', 'ambient_light', 'directional_light_color'.
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

    print(f"[Lighting {config_id}: {config['name']}] Task: {env_id}")
    print(f"  Ambient: {config['ambient_light']}, Directional: {config['directional_light_color']}")

    env.reset(**reset_kwargs)

    # Restore the initial environment state from the trajectory
    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    env.unwrapped.set_state_dict(ori_env_states[0])

    # --- Apply lighting color ---
    apply_lighting(env, config["ambient_light"], config["directional_light_color"])
    print(f"  Lighting applied: {config['name']}")

    # --- Replay and record ---
    ori_actions = h5_file[traj_id]["actions"][:]
    print(f"  Replaying {len(ori_actions)} actions...")

    frames = [env.render().cpu().numpy()[0]]
    for i, a in enumerate(ori_actions):
        env.step(a)
        # Restore exact env state from trajectory to prevent physics divergence
        env.unwrapped.set_state_dict(ori_env_states[i + 1])
        frames.append(env.render().cpu().numpy()[0])
        if i % 10 == 0:
            print(f"  Stepping {i}")

    # --- Save video ---
    out_dir = os.path.join(os.path.dirname(h5_path), "lighting_replay")
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"ep{episode['episode_id']}_light{config_id}.mp4")
    imageio.mimsave(out_video, frames, fps=env.unwrapped.control_freq)
    env.close()
    h5_file.close()
    print(f"  Done! Saved to {out_video}")


def main():
    parser = argparse.ArgumentParser(description="Replay trajectory with lighting domain randomization")
    parser.add_argument("--config-json", type=str, default="lighting_configs.json",
                        help="Path to lighting_configs.json")
    parser.add_argument("--config-id", type=str, required=True,
                        help="Lighting config ID (0-7) or 'all' to run all configs")
    parser.add_argument("--h5-path", type=str,
                        default="/data1/maniskill/downloaded_demos/PickCube-v1/motionplanning/trajectory.h5",
                        help="Path to trajectory HDF5 file")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="Episode index to replay")
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        all_configs = json.load(f)

    if args.config_id == "all":
        config_ids = sorted(all_configs.keys(), key=int)
    else:
        config_ids = [args.config_id]

    for config_key in config_ids:
        if config_key not in all_configs:
            raise ValueError(f"Config ID {config_key} not found. Valid range: 0-{len(all_configs)-1}")
        config = all_configs[config_key]
        replay_with_lighting(int(config_key), config, args.h5_path, args.episode_idx)


if __name__ == "__main__":
    main()
