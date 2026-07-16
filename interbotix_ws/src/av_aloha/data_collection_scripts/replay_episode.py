from pathlib import Path
import argparse
import time

import numpy as np
import rospy
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from data_col_config import ARM_MODES, ACTION_LAYOUTS
from arm_config import ARM_CONFIG

from robot_control import (
    create_and_configure_robot,
    stop_robots,
    replay_arm_command,
    reset_arm,
)

from gripper import command_gripper

def to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x

def extract_episode_indices(dataset, episode_idx):
    episode_data_index = dataset.episode_data_index
    start = int(episode_data_index["from"][episode_idx])
    end = int(episode_data_index["to"][episode_idx])
    return start, end

def parse_action(action, mode):
    layout = ACTION_LAYOUTS[mode]

    parsed = {}

    for key, idx in layout.items():
        parsed[key] = action[idx]

    return parsed

def get_reference_timestamp(sample, mode):
    arms = ARM_MODES[mode]

    first_arm = arms[0]

    return float(to_numpy_1d(sample[f"observation.timestamps.{first_arm}"])[0])

def print_sample_summary(sample, mode):
    print("\n===== SAMPLE =====")

    print("action:")
    print(to_numpy_1d(sample["action"]))

    print("\nobservation.state:")
    print(to_numpy_1d(sample["observation.state"]))

    for arm in ARM_MODES[mode]:

        ee_key = f"observation.ee_pose.{arm}"

        if ee_key in sample:
            print(f"\n{arm} ee:")
            print(to_numpy_1d(sample[ee_key]))

        ts_key = f"observation.timestamps.{arm}"

        if ts_key in sample:
            print(f"{arm} ts:")
            print(to_numpy_1d(sample[ts_key]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "left",
            "right",
            "middle",
            "bimanual",
            "all",
        ],
        default="right",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument(
        "--base-root",
        type=str,
        default="/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name, e.g. screwdriver_insertion")
    parser.add_argument("--run", type=str, default="latest", help="Run folder name or 'latest'")
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-offset", type=int, default=0)
    args = parser.parse_args()

    rospy.init_node("replay_episode", anonymous=True)

    robots = {
        arm_name: create_and_configure_robot(arm_name)
        for arm_name in ARM_MODES[args.mode]
    }

    stop_requested = False

    def request_stop():
        nonlocal stop_requested
        stop_requested = True
        stop_robots(robots)

    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
    else:
        if args.task is None:
            raise ValueError("Either --dataset-root OR (--task and optionally --run) must be provided")

        base_root = Path(args.base_root).expanduser().resolve()
        task_dir = base_root / args.task

        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

        if args.run == "latest":
            run_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                raise FileNotFoundError(f"No runs found under task directory: {task_dir}")
            dataset_root = sorted(run_dirs)[-1]
        else:
            dataset_root = task_dir / args.run

        dataset_root = dataset_root.resolve()

    print(f"Using dataset_root={dataset_root}")

    dataset = LeRobotDataset(str(dataset_root), video_backend="pyav")

    if args.episode_idx < 0 or args.episode_idx >= len(dataset.episode_data_index["from"]):
        raise ValueError(f"episode_idx {args.episode_idx} out of range")

    start, end = extract_episode_indices(dataset, args.episode_idx)
    start = min(start + args.start_offset, end)
    total_steps = end - start
    print(f"{total_steps} number of steps in episode {args.episode_idx}")

    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    if total_steps <= 0:
        raise ValueError("No steps to replay")

    fps = args.fps if args.fps is not None else float(dataset.fps)

    print(f"dataset_root={dataset_root}")
    print(f"episode_idx={args.episode_idx}")
    print(f"start={start}, end={end}, replay_steps={total_steps}, fps={fps}")

    first = dataset[start]
    print("sample keys:", list(first.keys()))
    print("first action:", to_numpy_1d(first["action"]))
    print("first observation.state:", to_numpy_1d(first["observation.state"]))

    if args.dry_run:
        print("Dry run only; exiting before robot motion.")
        return

    stop_requested = False

    rospy.on_shutdown(request_stop)

    first_sample = dataset[start]
    t0_dataset = get_reference_timestamp(first_sample, args.mode)
    t0_wall = time.monotonic()

    try:
        for arm_name, bot in robots.items():
            reset_arm(bot, arm_name)
            # measured_q = np.array(bot.dxl.joint_states.position[:6], dtype=float)
            n = ARM_CONFIG[arm_name]["num_joints"]
            measured_q = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
            print(f"Measured {arm_name} joints after reset:", np.round(measured_q, 4))

        print("Starting replay. Press Ctrl+C to stop.")

        for i in range(total_steps):
            if stop_requested or rospy.is_shutdown():
                break

            sample = dataset[start + i]
            action = to_numpy_1d(sample["action"])

            expected_dim = max(
                idx.stop if isinstance(idx, slice)
                else idx + 1
                for idx in ACTION_LAYOUTS[args.mode].values()
            )

            if len(action) < expected_dim:
                raise ValueError(f"Expected action dim >= {expected_dim}. Got {len(action)}")

            parsed = parse_action(action, args.mode)

            sample_ts = get_reference_timestamp(sample, args.mode)
            target_wall_time = t0_wall + (sample_ts - t0_dataset)
            sleep_time = target_wall_time - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)

            if sleep_time < -0.01:
                print(f"WARNING: replay lagging by {-sleep_time*1000:.1f} ms")

            for arm_name, bot in robots.items():

                arm_key = f"{arm_name}_arm"

                if arm_key not in parsed:
                    continue

                cmd = np.asarray(parsed[arm_key], dtype=np.float32)

                if np.any(np.isnan(cmd)):
                    print(f"{arm_name}: NaN command detected")
                    continue

                expected_joints = 7 if arm_name == "middle" else 6

                if len(cmd) != expected_joints:
                    print(f"{arm_name}: invalid command length {len(cmd)}")
                    continue

                replay_arm_command(bot, cmd)

                gripper_key = f"{arm_name}_gripper"

                if gripper_key in parsed:
                    command_gripper(bot, parsed[gripper_key])
                    print(f"{arm_name} gripper={parsed[gripper_key]:.3f}")

            time.sleep(0.03)

            for arm_name, bot in robots.items():

                arm_key = f"{arm_name}_arm"

                if arm_key not in parsed:
                    continue

                # measured_q = np.asarray(bot.dxl.joint_states.position[:6], dtype=np.float32)

                n = ARM_CONFIG[arm_name]["num_joints"]
                measured_q = np.asarray(bot.dxl.joint_states.position[:n], dtype=np.float32)

                print(f"{arm_name} target={np.round(parsed[arm_key], 3)}")
                print(f"{arm_name} measured={np.round(measured_q, 3)}")

                tracking_error = np.linalg.norm(measured_q - parsed[arm_key])

                print(f"{arm_name} tracking_error={tracking_error:.4f}")

    except KeyboardInterrupt:
        request_stop()
    finally:
        stop_robots(robots)
        # for arm_name, bot in robots.items():
        #     reset_arm(bot, arm_name)

if __name__ == "__main__":
    main()