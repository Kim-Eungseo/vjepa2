import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import sapien
import torch
from tqdm import tqdm

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6RobotiqMotionPlanningSolver,
)
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.utils.wrappers.record import RecordEpisode


class CustomXArm6RobotiqMotionPlanningSolver(XArm6RobotiqMotionPlanningSolver):
    """XArm6 solver with tighter joint limits and configurable early-stop on success."""

    def setup_planner(self):
        planner = super().setup_planner()
        limits = np.asarray(planner.joint_limits)
        for idx in (0, 3, 5):
            limits[idx][0] = max(limits[idx][0], -np.pi)
            limits[idx][1] = min(limits[idx][1], np.pi)
        planner.joint_limits = limits
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        success_step = None
        post_success_steps = int(getattr(self, "post_success_steps", 0))
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()

            # Stop on success by default. Optionally allow a few post-success steps
            # for smoother endings on tasks like PushCube.
            if getattr(self, "stop_on_success", True) and info and _to_bool(info.get("success", False)):
                if success_step is None:
                    success_step = self.elapsed_steps
                if (self.elapsed_steps - success_step) >= post_success_steps:
                    break

        return obs, reward, terminated, truncated, info


def _to_bool(v) -> bool:
    if isinstance(v, torch.Tensor):
        return bool(v.item())
    return bool(v)


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


def _move_with_fallback(planner, pose, dry_run: bool = False, refine_steps: int = 0):
    res = planner.move_to_pose_with_screw(
        pose,
        dry_run=dry_run,
        refine_steps=refine_steps,
    )
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(
            pose,
            dry_run=dry_run,
            refine_steps=refine_steps,
        )
    return res


def _build_planner(env, debug=False, vis=False, joint_vel_limits=0.9, joint_acc_limits=0.9):
    return CustomXArm6RobotiqMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=joint_vel_limits,
        joint_acc_limits=joint_acc_limits,
    )


def solve_pick_cube_xarm6(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel")

    planner = _build_planner(env, debug=debug, vis=vis)
    try:
        env_u = env.unwrapped
        obb = get_actor_obb(env_u.cube)
        approaching = np.array([0, 0, -1])
        target_closing = (
            env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )

        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        grasp_pose = env_u.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            env_u.cube.pose.sp.p,
        )

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = _move_with_fallback(planner, reach_pose)
        if res == -1:
            return -1

        res = _move_with_fallback(planner, grasp_pose)
        if res == -1:
            return -1

        planner.close_gripper()

        goal_pose = sapien.Pose(env_u.goal_site.pose.sp.p, grasp_pose.q)
        res = _move_with_fallback(planner, goal_pose)
        return res
    finally:
        planner.close()


def solve_stack_cube_xarm6(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel")

    from transforms3d.euler import euler2quat

    planner = _build_planner(env, debug=debug, vis=vis)
    try:
        env_u = env.unwrapped
        obb = get_actor_obb(env_u.cubeA)
        approaching = np.array([0, 0, -1])
        target_closing = (
            env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )

        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        grasp_pose = env_u.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            grasp_info["center"],
        )

        angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
        angles = np.repeat(angles, 2)
        angles[1::2] *= -1
        for angle in angles:
            candidate = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle))
            if _move_with_fallback(planner, candidate, dry_run=True) != -1:
                grasp_pose = candidate
                break

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = _move_with_fallback(planner, reach_pose)
        if res == -1:
            return -1

        res = _move_with_fallback(planner, grasp_pose)
        if res == -1:
            return -1
        planner.close_gripper()

        lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
        res = _move_with_fallback(planner, lift_pose)
        if res == -1:
            return -1

        goal_pose = env_u.cubeB.pose * sapien.Pose([0, 0, env_u.cube_half_size[2] * 2])
        offset = (goal_pose.p - env_u.cubeA.pose.p).cpu().numpy()[0]
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        res = _move_with_fallback(planner, align_pose)
        if res == -1:
            return -1

        return planner.open_gripper()
    finally:
        planner.close()


def solve_push_cube_xarm6(env, seed=None, debug=False, vis=False):
    """PushCube tuned for xarm6: approach from -x side and use motion fallback."""

    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel")

    planner = _build_planner(env, debug=debug, vis=vis)
    # Allow a short tail after success, then stop to avoid long post-success drift.
    planner.stop_on_success = True
    planner.post_success_steps = 1
    try:
        env_u = env.unwrapped
        # Push contact can pry Robotiq fingers open near the goal. Increase
        # gripper drive force for this task to better preserve a closed shape.
        gripper_active = env_u.agent.controller.controllers.get("gripper_active", None)
        if gripper_active is not None:
            gripper_active.config.force_limit = 5.0
            gripper_active.set_drive_property()

        planner.close_gripper()

        tcp_q = env_u.agent.tcp.pose.sp.q
        contact_offset = np.array([-0.05, 0, 0])

        reach_pose = sapien.Pose(p=env_u.obj.pose.sp.p + contact_offset, q=tcp_q)
        res = _move_with_fallback(planner, reach_pose)
        if res == -1:
            return -1

        goal_pose = sapien.Pose(p=env_u.goal_region.pose.sp.p + contact_offset, q=tcp_q)
        res = _move_with_fallback(planner, goal_pose)
        return res
    finally:
        planner.close()


def solve_pull_cube_xarm6(env, seed=None, debug=False, vis=False):
    """PullCube tuned for xarm6: approach from +x side and pull to goal region."""

    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel")

    planner = _build_planner(env, debug=debug, vis=vis)
    # PullCube has a lenient success region; run through the full planned segment
    # to better match Panda-style trajectories.
    planner.stop_on_success = False
    try:
        env_u = env.unwrapped
        planner.close_gripper()

        tcp_q = env_u.agent.tcp.pose.sp.q
        pull_offset = np.array([0.05, 0, 0])

        reach_pose = sapien.Pose(p=env_u.obj.pose.sp.p + pull_offset, q=tcp_q)
        res = _move_with_fallback(planner, reach_pose)
        if res == -1:
            return -1

        goal_pose = sapien.Pose(p=env_u.goal_region.pose.sp.p + pull_offset, q=tcp_q)
        res = _move_with_fallback(planner, goal_pose)
        return res
    finally:
        planner.close()


def solve_lift_peg_upright_xarm6(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel")

    planner = _build_planner(
        env,
        debug=debug,
        vis=vis,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    try:
        env_u = env.unwrapped
        obb = get_actor_obb(env_u.peg)
        approaching = np.array([0, 0, -1])
        target_closing = (
            env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )

        # 0.0 is empirically more stable on xarm6 than the Panda default for this task.
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.0,
        )
        base_grasp_pose = env_u.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            grasp_info["center"],
        )
        grasp_pose = base_grasp_pose * sapien.Pose([0.10, 0, 0])

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = _move_with_fallback(planner, reach_pose)
        if res == -1:
            return -1

        res = _move_with_fallback(planner, grasp_pose)
        if res == -1:
            return -1
        planner.close_gripper()

        lift_pose = sapien.Pose([0, 0, 0.30]) * grasp_pose
        res = _move_with_fallback(planner, lift_pose)
        if res == -1:
            return -1

        theta = np.pi / 10
        rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])
        final_pose = lift_pose * sapien.Pose(p=[0, 0, 0], q=rotation_quat)

        res = _move_with_fallback(planner, final_pose)
        if res == -1:
            return -1

        lower_pose = sapien.Pose([0, 0, -0.10]) * final_pose
        res = _move_with_fallback(planner, lower_pose)
        return res
    finally:
        planner.close()


@contextmanager
def _patched_attr(module, attr: str, value):
    old = getattr(module, attr)
    setattr(module, attr, value)
    try:
        yield
    finally:
        setattr(module, attr, old)


def solve_place_sphere_experimental(env, seed=None, debug=False, vis=False):
    import mani_skill.examples.motionplanning.panda.solutions.place_sphere as pb_place_sphere

    with _patched_attr(
        pb_place_sphere,
        "PandaArmMotionPlanningSolver",
        CustomXArm6RobotiqMotionPlanningSolver,
    ):
        return pb_place_sphere.solve(env, seed=seed, debug=debug, vis=vis)


def solve_stack_pyramid_experimental(env, seed=None, debug=False, vis=False):
    import mani_skill.examples.motionplanning.panda.solutions.stack_pyramid as pb_stack_pyramid

    with _patched_attr(
        pb_stack_pyramid,
        "PandaArmMotionPlanningSolver",
        CustomXArm6RobotiqMotionPlanningSolver,
    ):
        return pb_stack_pyramid.solve(env, seed=seed, debug=debug, vis=vis)


def solve_pull_cube_tool_xarm6(env, seed=None, debug=False, vis=False):
    """
    PullCubeTool xarm6 policy that explicitly grasps and uses the tool.
    This task is not officially xarm6-supported, so we tune gripper force and
    require tool-grasp success before attempting hook-and-pull.
    """

    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel")

    planner = _build_planner(
        env,
        debug=debug,
        vis=vis,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    # Avoid saving trivial 1-step trajectories when success is already true at
    # reset and stop shortly after success is reached.
    planner.stop_on_success = True
    planner.post_success_steps = 1

    try:
        env_u = env.unwrapped
        init_info = env_u.evaluate()
        if _to_bool(init_info.get("success", False)):
            return -1

        # Heavier tool needs stronger gripper drive to keep grasp stable.
        gripper_active = env_u.agent.controller.controllers.get("gripper_active", None)
        if gripper_active is not None:
            gripper_active.config.force_limit = 8.0
            gripper_active.set_drive_property()

        tool_obb = get_actor_obb(env_u.l_shape_tool)
        approaching = np.array([0, 0, -1])
        target_closing = (
            env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )

        grasp_info = compute_grasp_info_by_obb(
            tool_obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.03,
        )
        grasp_pose = env_u.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            env_u.l_shape_tool.pose.sp.p,
        )
        grasp_pose = grasp_pose * sapien.Pose([0.02, 0, 0])

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = _move_with_fallback(planner, reach_pose)
        if res == -1:
            return -1

        res = _move_with_fallback(planner, grasp_pose)
        if res == -1:
            return -1

        planner.close_gripper(t=12)
        # Brief extra hold to reduce post-grasp wobble before lifting.
        planner.close_gripper(t=4)
        if not _to_bool(env_u.agent.is_grasping(env_u.l_shape_tool, max_angle=30)[0]):
            return -1

        # Keep the transport height lower than before while still clearing cube/tool.
        lift_height = 0.30
        lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
        lift_pose.set_q(grasp_pose.q)
        res = _move_with_fallback(planner, lift_pose)
        if res == -1:
            return -1

        cube_pos = env_u.cube.pose.sp.p
        # Stage 1: approach with extra longitudinal clearance.
        approach_offset = sapien.Pose(
            [-(env_u.hook_length + env_u.cube_half_size + 0.12), 0.0, lift_height - 0.06]
        )
        approach_pose = sapien.Pose(cube_pos) * approach_offset
        approach_pose.set_q(grasp_pose.q)
        res = _move_with_fallback(planner, approach_pose)
        if res == -1:
            return -1

        # Stage 2: descend near hook line with room before final contact.
        pre_hook_offset = sapien.Pose(
            [-(env_u.hook_length + env_u.cube_half_size + 0.06), -0.067, 0.10]
        )
        pre_hook_pose = sapien.Pose(cube_pos) * pre_hook_offset
        pre_hook_pose.set_q(grasp_pose.q)
        res = _move_with_fallback(planner, pre_hook_pose)
        if res == -1:
            return -1

        # Stage 3: final hook placement slightly above table for safer contact.
        hook_offset = sapien.Pose(
            [-(env_u.hook_length + env_u.cube_half_size + 0.02), -0.067, 0.02]
        )
        hook_pose = sapien.Pose(cube_pos) * hook_offset
        hook_pose.set_q(grasp_pose.q)
        res = _move_with_fallback(planner, hook_pose)
        if res == -1:
            return -1

        target_pose = hook_pose * sapien.Pose([-0.35, 0, 0])
        return _move_with_fallback(planner, target_pose)
    finally:
        planner.close()


def solve_plug_charger_experimental(env, seed=None, debug=False, vis=False):
    import mani_skill.examples.motionplanning.xarm6.solutions.plug_charger as xb_plug_charger

    with _patched_attr(
        xb_plug_charger,
        "XArm6RobotiqMotionPlanningSolver",
        CustomXArm6RobotiqMotionPlanningSolver,
    ):
        return xb_plug_charger.solve(env, seed=seed, debug=debug, vis=vis)


TaskSolver = Callable[..., object]

STABLE_TASK_SOLVERS: "OrderedDict[str, TaskSolver]" = OrderedDict(
    {
        "PickCube-v1": solve_pick_cube_xarm6,
        "StackCube-v1": solve_stack_cube_xarm6,
        "PushCube-v1": solve_push_cube_xarm6,
        "PullCube-v1": solve_pull_cube_xarm6,
        "LiftPegUpright-v1": solve_lift_peg_upright_xarm6,
    }
)

EXPERIMENTAL_TASK_SOLVERS: "OrderedDict[str, TaskSolver]" = OrderedDict(
    {
        "PlaceSphere-v1": solve_place_sphere_experimental,
        "StackPyramid-v1": solve_stack_pyramid_experimental,
        "PullCubeTool-v1": solve_pull_cube_tool_xarm6,
        "PlugCharger-v1": solve_plug_charger_experimental,
    }
)

DISABLED_TASKS: Dict[str, str] = {
    "PegInsertionSide-v1": "env initialization fails with xarm6_robotiq (DOF mismatch) in this ManiSkill build.",
}

ALL_TASK_SOLVERS: "OrderedDict[str, TaskSolver]" = OrderedDict(STABLE_TASK_SOLVERS)
ALL_TASK_SOLVERS.update(EXPERIMENTAL_TASK_SOLVERS)


def _solver_registry(task_set: str) -> "OrderedDict[str, TaskSolver]":
    if task_set == "stable":
        return OrderedDict(STABLE_TASK_SOLVERS)
    if task_set == "experimental":
        return OrderedDict(EXPERIMENTAL_TASK_SOLVERS)
    if task_set == "all":
        return OrderedDict(ALL_TASK_SOLVERS)
    raise ValueError(f"Unknown task set: {task_set}")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PickCube-v1",
        help=(
            "Environment to run. Use a concrete task id (e.g. PickCube-v1) or 'all' "
            "to run multiple tasks from --task-set."
        ),
    )
    parser.add_argument(
        "--task-set",
        choices=["stable", "experimental", "all"],
        default="stable",
        help="Task subset used only when --env-id all.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available stable/experimental/disabled tasks and exit.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="When --env-id all, stop immediately if one task fails.",
    )
    parser.add_argument(
        "-o",
        "--obs-mode",
        type=str,
        default="none",
        help="Observation mode used during base trajectory collection.",
    )
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument(
        "--only-count-success",
        action="store_true",
        help="Collect until num_traj successful trajectories are saved.",
    )
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument(
        "-b",
        "--sim-backend",
        type=str,
        default="auto",
        help="Simulation backend: auto/cpu/gpu.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        help="Render mode used for video output.",
    )
    parser.add_argument("--vis", action="store_true", help="Open GUI visualization.")
    parser.add_argument("--save-video", action="store_true", help="Save videos locally.")
    parser.add_argument("--traj-name", type=str, help="Trajectory filename (.h5 stem).")
    parser.add_argument(
        "--shader",
        default="default",
        type=str,
        help="Shader pack for rendering (default/rt/rt-fast).",
    )
    parser.add_argument("--record-dir", type=str, default="demos", help="Output root directory.")
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="CPU multiprocessing workers per task.",
    )
    return parser.parse_args(args=args)


def _print_task_table() -> None:
    print("Stable tasks (default):")
    for env_id in STABLE_TASK_SOLVERS:
        print(f"  - {env_id}")
    print("\nExperimental tasks (optional, lower reliability):")
    for env_id in EXPERIMENTAL_TASK_SOLVERS:
        print(f"  - {env_id}")
    print("\nDisabled tasks (known hard failure):")
    for env_id, reason in DISABLED_TASKS.items():
        print(f"  - {env_id}: {reason}")


def _new_traj_name(args, proc_id: int) -> str:
    name = args.traj_name or time.strftime("%Y%m%d_%H%M%S")
    if args.num_procs > 1:
        name = f"{name}.{proc_id}"
    return name


def _collect_worker(args, env_id: str, proc_id: int = 0, start_seed: int = 0) -> str:
    solver = ALL_TASK_SOLVERS[env_id]
    make_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        robot_uids="xarm6_robotiq",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )
    # If reward mode is not explicitly requested, defer to each task's default.
    # Some tabletop tasks (e.g., PlugCharger-v1 in this build) do not support "dense".
    if args.reward_mode is not None:
        make_kwargs["reward_mode"] = args.reward_mode

    env = gym.make(
        env_id,
        **make_kwargs,
    )

    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=_new_traj_name(args, proc_id),
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="xarm6 tabletop motion planning collection",
        video_fps=30,
        save_on_reset=False,
    )

    output_h5_path = env._h5_file.filename
    print(f"Motion Planning Running on {env_id} with xarm6_robotiq")

    pbar = tqdm(total=args.num_traj, desc=f"{env_id} | proc_id: {proc_id}")
    seed = start_seed
    passed = 0
    attempted = 0
    failed_motion_plans = 0
    successes = []
    episode_lengths = []

    while passed < args.num_traj:
        attempted += 1
        try:
            res = solver(env, seed=seed, debug=False, vis=bool(args.vis))
        except Exception as exc:
            print(f"[{env_id}] motion planning solver exception @ seed={seed}: {exc}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            info = res[-1]
            success = _to_bool(info.get("success", False))
            elapsed = _to_int(info.get("elapsed_steps", None))
            if elapsed is not None:
                episode_lengths.append(elapsed)

        successes.append(success)

        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue

        env.flush_trajectory(save=True)
        if args.save_video:
            env.flush_video(save=True)

        passed += 1
        seed += 1
        pbar.update(1)
        pbar.set_postfix(
            dict(
                success_rate=float(np.mean(successes)) if successes else 0.0,
                failed_motion_plan_rate=(failed_motion_plans / attempted) if attempted else 0.0,
                avg_episode_length=(float(np.mean(episode_lengths)) if episode_lengths else 0.0),
                max_episode_length=(int(np.max(episode_lengths)) if episode_lengths else 0),
            )
        )

    pbar.close()
    env.close()
    return output_h5_path


def _merged_h5_path(example_h5_path: str) -> str:
    stem, ext = osp.splitext(example_h5_path)
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        stem = parts[0]
    return stem + ext


def _run_task(args, env_id: str) -> Optional[str]:
    if args.num_procs <= 1:
        return _collect_worker(args, env_id)

    if args.num_traj < args.num_procs:
        raise ValueError("num_traj must be >= num_procs when multiprocessing is enabled")

    base = args.num_traj // args.num_procs
    rem = args.num_traj % args.num_procs

    proc_args = []
    next_seed = 0
    for proc_id in range(args.num_procs):
        count = base + (1 if proc_id < rem else 0)
        if count <= 0:
            continue
        args_i = deepcopy(args)
        args_i.num_traj = count
        proc_args.append((args_i, env_id, proc_id, next_seed))
        next_seed += count

    with mp.Pool(len(proc_args)) as pool:
        h5_paths = pool.starmap(_collect_worker, proc_args)

    if len(h5_paths) == 1:
        return h5_paths[0]

    output_path = _merged_h5_path(h5_paths[0])
    merge_trajectories(output_path, h5_paths)

    for h5_path in h5_paths:
        tqdm.write(f"Remove {h5_path}")
        if osp.exists(h5_path):
            os.remove(h5_path)
        json_path = h5_path.replace(".h5", ".json")
        tqdm.write(f"Remove {json_path}")
        if osp.exists(json_path):
            os.remove(json_path)

    return output_path


def main(args):
    if args.list_tasks:
        _print_task_table()
        return

    if args.env_id in DISABLED_TASKS:
        raise RuntimeError(f"{args.env_id} is disabled: {DISABLED_TASKS[args.env_id]}")

    if args.env_id == "all":
        solver_map = _solver_registry(args.task_set)
        if not solver_map:
            raise RuntimeError(f"No tasks found for task-set={args.task_set}")

        summary = []
        for env_id in solver_map.keys():
            if env_id in DISABLED_TASKS:
                tqdm.write(f"[SKIP] {env_id}: {DISABLED_TASKS[env_id]}")
                summary.append((env_id, "skipped", "disabled"))
                continue

            tqdm.write(f"\n===== Start {env_id} =====")
            try:
                out = _run_task(args, env_id)
                summary.append((env_id, "ok", out or ""))
            except Exception as exc:
                summary.append((env_id, "failed", str(exc)))
                tqdm.write(f"[FAILED] {env_id}: {exc}")
                if args.fail_fast:
                    raise

        print("\n===== Collection Summary =====")
        for env_id, status, info in summary:
            print(f"{env_id:20s} | {status:7s} | {info}")
        return

    if args.env_id not in ALL_TASK_SOLVERS:
        available = list(ALL_TASK_SOLVERS.keys()) + ["all"]
        raise RuntimeError(f"Unknown env-id {args.env_id}. Available: {available}")

    _run_task(args, args.env_id)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
