import os
import h5py
import json
import argparse
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import wrappers, io_utils
from sapien.core import Pose
from transforms3d.quaternions import mat2quat

DEFAULT_CAMERAS = {
    "PickCube-v1": {"eye": [0.6, 0.7, 0.6], "target": [0.0, 0.0, 0.35]},
    "PegInsertionSide-v1": {"eye": [0.0, -0.3, 0.2], "target": [0.0, 0.0, 0.1]},
}

def look_at(eye, target, up=np.array([0, 0, 1])):
    forward = np.array(target) - np.array(eye)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0, 1, 0])
    else:
        right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.stack([forward, -right, up], axis=1) # Sapien convention is x: forward, y: left, z: up
    quat = mat2quat(mat)
    return Pose(eye, quat)

def get_new_pose(original_eye, original_target, az_delta_deg, el_delta_deg, dist_scale):
    original_eye = np.array(original_eye)
    original_target = np.array(original_target)
    
    vec = original_eye - original_target
    original_radius = np.linalg.norm(vec)
    
    if original_radius < 1e-6:
        original_radius = 1.0
        original_azimuth = 0.0
        original_elevation = 0.0
    else:
        original_azimuth = np.arctan2(vec[1], vec[0])
        original_elevation = np.arcsin(np.clip(vec[2] / original_radius, -1.0, 1.0))
        
    az_delta_rad = np.radians(az_delta_deg)
    el_delta_rad = np.radians(el_delta_deg)
    
    new_azimuth = original_azimuth + az_delta_rad
    new_elevation = original_elevation + el_delta_rad
    
    # Clip elevation to avoid exact vertical which flips the look-at calculation
    new_elevation = np.clip(new_elevation, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    
    new_radius = original_radius * dist_scale
    
    dx = new_radius * np.cos(new_elevation) * np.cos(new_azimuth)
    dy = new_radius * np.cos(new_elevation) * np.sin(new_azimuth)
    dz = new_radius * np.sin(new_elevation)
    
    new_eye = original_target + np.array([dx, dy, dz])
    
    try:
        from mani_skill.utils.sapien_utils import look_at as ms_look_at
        return ms_look_at(new_eye, original_target)
    except:
        return look_at(new_eye, original_target)

def main():
    parser = argparse.ArgumentParser(description="Replay trajectory with domain randomized camera poses")
    parser.add_argument("--traj-path", type=str, default="/data1/maniskill/downloaded_demos/PickCube-v1/motionplanning/trajectory.h5", help="Path to trajectory .h5 file")
    parser.add_argument("--config-id", type=int, required=True, help="ID of the camera configuration (0-299)")
    parser.add_argument("--config-file", type=str, default="camera_configs.json", help="Path to the JSON config file")
    parser.add_argument("--task", type=str, default=None, help="The task/env ID. If not provided, inferred from trajectory json.")
    parser.add_argument("--episode-idx", type=int, default=0, help="Index of the episode to replay (default: 0)")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        camera_configs = json.load(f)
        
    if str(args.config_id) not in camera_configs:
        raise ValueError(f"Config ID {args.config_id} not found in {args.config_file}")
        
    cam_config = camera_configs[str(args.config_id)]
    
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    env_kwargs = json_data["env_info"]["env_kwargs"]
    
    task_id = args.task if args.task else json_data["env_info"]["env_id"]
    
    print(f"Auto-detecting default camera pose from {task_id}...")
    tmp_kwargs = env_kwargs.copy()
    if "human_render_camera_configs" in tmp_kwargs:
        del tmp_kwargs["human_render_camera_configs"]
    if "num_envs" in tmp_kwargs:
        tmp_kwargs["num_envs"] = 1
    
    # Instantiate a temporary environment to read defaults
    tmp_env = gym.make(task_id, **tmp_kwargs)
    
    try:
        cam_configs = tmp_env.unwrapped._default_human_render_camera_configs
        if isinstance(cam_configs, (list, tuple)) and len(cam_configs) > 0:
            cam_config_obj = cam_configs[0]
        elif isinstance(cam_configs, dict):
            cam_config_obj = list(cam_configs.values())[0]
        else:
            cam_config_obj = cam_configs
            
        # Convert SAPIEN tensors to numpy
        p = cam_config_obj.pose.p.cpu().numpy()[0]
        q = cam_config_obj.pose.q.cpu().numpy()[0]
        
        original_eye = p
        
        from transforms3d.quaternions import quat2mat
        mat = quat2mat(q)
        look_dir = mat[:, 0]  # SAPIEN forward is +x
        
        denom = look_dir[0]**2 + look_dir[1]**2
        if denom > 1e-6:
            t = -(look_dir[0]*original_eye[0] + look_dir[1]*original_eye[1]) / denom
        else:
            if abs(look_dir[2]) > 1e-6:
                t = -original_eye[2] / look_dir[2]
            else:
                t = 1.0  # fallback
        
        original_target = original_eye + t * look_dir
        print(f"Auto-detected eye: {original_eye}, target: {original_target}")
    except Exception as e:
        print(f"Warning: Could not auto-detect camera ({e}). Using fallback defaults.")
        defaults = DEFAULT_CAMERAS.get(task_id, {"eye": [0.6, 0.7, 0.6], "target": [0.0, 0.0, 0.35]})
        original_eye = defaults["eye"]
        original_target = defaults["target"]
        print(f"Using fallback eye: {original_eye}, target: {original_target}")
        
    tmp_env.close()
        
    pose = get_new_pose(
        original_eye, 
        original_target, 
        cam_config["azimuth_delta"], 
        cam_config["elevation_delta"], 
        cam_config["distance_scale"]
    )
    
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["human_render_camera_configs"] = {"render_camera": {"pose": pose}}
    
    env_kwargs["sim_backend"] = "physx_cpu"
    if "num_envs" in env_kwargs:
        env_kwargs["num_envs"] = 1
        
    env = gym.make(task_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)
        
    h5_file = h5py.File(args.traj_path, "r")
    episodes = json_data["episodes"]
    
    if args.episode_idx >= len(episodes):
        raise ValueError(f"Episode index {args.episode_idx} is out of bounds (max {len(episodes)-1})")
        
    episode = episodes[args.episode_idx]
    traj_id = f"traj_{episode['episode_id']}"
    
    reset_kwargs = episode["reset_kwargs"].copy()
    if "seed" in reset_kwargs and isinstance(reset_kwargs["seed"], list):
        reset_kwargs["seed"] = reset_kwargs["seed"][0]
        
    print(f"Starting replay for task={task_id}, episode_idx={args.episode_idx}, episode_id={episode['episode_id']}...")
    env.reset(**reset_kwargs)
    
    from mani_skill.trajectory.utils import dict_to_list_of_dicts
    ori_env_states = dict_to_list_of_dicts(h5_file[traj_id]["env_states"])
    env.unwrapped.set_state_dict(ori_env_states[0])
    
    ori_actions = h5_file[traj_id]["actions"][:]
    print(f"Number of actions to replay: {len(ori_actions)}")
    
    import imageio
    frames = [env.render().cpu().numpy()[0]]
    for i, a in enumerate(ori_actions):
        env.step(a)
        env.unwrapped.set_state_dict(ori_env_states[i + 1])
        frames.append(env.render().cpu().numpy()[0])
        if i % 10 == 0:
            print(f"Stepping {i}/{len(ori_actions)}")

    out_dir = os.path.join(os.path.dirname(args.traj_path), "cam_replay")
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"ep{episode['episode_id']}_cam{args.config_id}.mp4")
    
    print(f"Saving video to {out_video}...")
    imageio.mimsave(out_video, frames, fps=env.unwrapped.control_freq)
    
    env.close()
    h5_file.close()
    print("Done!")

if __name__ == "__main__":
    main()
