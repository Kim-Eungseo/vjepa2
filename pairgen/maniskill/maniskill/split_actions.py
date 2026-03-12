"""Split merged ManiSkill trajectory h5 into per-episode h5/json files.

For each episode, extracts actions (and optionally env_states) from the
merged h5 file and saves to episode_N/actions/ directory.

Usage:
    python split_actions.py \
        --traj-path /data1/maniskill/demos/LiftPegUpright-v1/motionplanning/joint_delta_pos.h5 \
        --output-dir /data1/maniskill/datasets/LiftPegUpright-v1 \
        --control-mode joint_delta_pos \
        --count 2
"""
import os
import json
import argparse
import h5py
import numpy as np
from mani_skill.utils import io_utils


def split_single_episode(h5_file, episode, control_mode, output_dir):
    """Extract a single episode from the merged h5 and save per-episode files."""
    episode_id = episode["episode_id"]
    traj_id = f"traj_{episode_id}"
    ep_dir = os.path.join(output_dir, f"ep{episode_id:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    # Save per-episode h5 using h5py's native copy
    h5_path = os.path.join(ep_dir, f"{control_mode}.h5")
    with h5py.File(h5_path, "w") as out_h5:
        h5_file.copy(traj_id, out_h5)

    # Save per-episode json
    json_path = os.path.join(ep_dir, f"{control_mode}.json")
    ep_meta = {
        "episode_id": episode_id,
        "episode_seed": episode.get("episode_seed"),
        "control_mode": episode.get("control_mode", control_mode),
        "elapsed_steps": episode.get("elapsed_steps"),
        "reset_kwargs": episode.get("reset_kwargs"),
        "success": episode.get("success"),
    }
    with open(json_path, "w") as f:
        json.dump(ep_meta, f, indent=2)

    return episode_id


def main():
    parser = argparse.ArgumentParser(description="Split merged h5 into per-episode files")
    parser.add_argument("--traj-path", type=str, required=True,
                        help="Path to merged trajectory h5 file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output directory")
    parser.add_argument("--control-mode", type=str, required=True,
                        help="Control mode name for output files (e.g. joint_delta_pos, ee_delta_pose)")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of episodes to split (default: all)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting episode index (default: 0)")
    args = parser.parse_args()

    # Determine json path - try exact match first, then common patterns
    json_path = args.traj_path.replace(".h5", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON metadata not found: {json_path}")

    json_data = io_utils.load_json(json_path)
    episodes = json_data["episodes"]

    h5_file = h5py.File(args.traj_path, "r")

    start = args.start_idx
    end = len(episodes) if args.count is None else min(start + args.count, len(episodes))

    print(f"Splitting {args.traj_path}")
    print(f"  Control mode: {args.control_mode}")
    print(f"  Episodes: {start} to {end-1} ({end-start} total)")
    print(f"  Output: {args.output_dir}/episode_*/actions/{args.control_mode}.*")
    print()

    for idx in range(start, end):
        ep = episodes[idx]
        eid = split_single_episode(h5_file, ep, args.control_mode, args.output_dir)
        if (idx - start + 1) % 100 == 0 or idx == end - 1:
            print(f"  [{idx-start+1}/{end-start}] ep{eid:03d}")

    h5_file.close()
    print(f"\nDone! Split {end-start} episodes to {args.output_dir}/")


if __name__ == "__main__":
    main()
