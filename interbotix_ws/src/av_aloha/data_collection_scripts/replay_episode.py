"""Replay a recorded episode's ACTION stream back onto the arms.

WHAT IS BEING REPLAYED
======================
`action` holds the joint vector this rig actually COMMANDED at each tick, in
DRIVER coordinates -- what `set_joint_positions` received after IK, after both
clamps and after the collision gates.  So replay is a straight re-send: no IK,
no solver, no frames decoded.  `observation.state` (the measured joints) is
recorded alongside and is only ever used here to report tracking error.

THE MODE IS A PROPERTY OF THE DATASET, NOT OF THE COMMAND LINE
==============================================================
The action vector's layout depends on which arms were active when it was
recorded (ACTION_LAYOUTS in data_col_config.py).  Getting that wrong is not a
crash -- a three-arm recording read as `--mode right` slices indices 0..5,
which is the LEFT arm's joints, and sends them to the right arm.  So the mode
now comes from the dataset's own `meta.json`, and an explicit `--mode` that
disagrees with the recorded action width is refused rather than obeyed.

TIMING: THE EPISODE IS READ UP FRONT
=====================================
`dataset[i]` decodes that frame's video for every camera.  Doing that inside a
50 Hz send loop cannot hold the rate, so the whole episode's actions and
timestamps are pulled from the non-video columns before any motion starts and
the send loop touches nothing but numpy.  (The previous version also slept a
fixed 30 ms per iteration ON TOP of pacing to the recorded timestamps, which
alone put a 50 Hz recording at roughly two-thirds speed and printed a
lag warning on every tick.)

STARTING POSITION
=================
The arms are reset to `forward`, which is nowhere near where an arbitrary
episode begins.  interbotix validates every command as
|goal - last_command| / moving_time against the joint velocity limit and
REFUSES the whole group command if any joint fails, so jumping straight into
the recording's first action gets silently dropped and the arm sits still
while replay runs away from it.  The first action is therefore approached with
the same step-limited interpolation the startup poses use.
"""

from pathlib import Path
import argparse
import json
import os
import time

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
try:
    import rospy
except ImportError:
    rospy = None

if __package__:
    from .data_col_config import ARM_MODES, ACTION_LAYOUTS, DATASET_ROOT
    from .arm_config import ARM_CONFIG
    from .robot_control import (
        create_and_configure_robot,
        stop_robots,
        replay_arm_command,
        reset_arm,
        interpolate_to_pose,
    )
    from .gripper import command_gripper
else:
    from data_col_config import ARM_MODES, ACTION_LAYOUTS, DATASET_ROOT
    from arm_config import ARM_CONFIG
    from robot_control import (
        create_and_configure_robot,
        stop_robots,
        replay_arm_command,
        reset_arm,
        interpolate_to_pose,
    )
    from gripper import command_gripper


## How close an arm must get to the episode's first commanded pose before the
## timed replay is allowed to start.  Generous: the point is to catch "did not
## move at all", not to grade the approach.
START_POSE_TOLERANCE = float(os.environ.get("GIAVA_REPLAY_START_TOL", "0.15"))


def to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x


def action_width(mode):
    """Number of values ACTION_LAYOUTS[mode] describes."""
    return max(
        idx.stop if isinstance(idx, slice) else idx + 1
        for idx in ACTION_LAYOUTS[mode].values()
    )


def read_dataset_meta(dataset_root):
    """The meta.json save_dataset_metadata() wrote next to the episodes.

    Absent for datasets recorded before it existed, so every caller treats a
    missing file as "unknown", not as an error."""
    path = Path(dataset_root) / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.load(open(path))
    except Exception as exc:
        print(f"[meta] could not read {path}: {exc}")
        return {}


def frame_table(dataset):
    """The episode/action/timestamp columns, WITHOUT decoding any video.

    `dataset[i]` pulls every camera's frame for that index.  The columns this
    replay needs all live in the parquet, which `hf_dataset` exposes directly;
    the fallback exists only for a LeRobotDataset that does not expose it."""
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None:
        print("[replay] no hf_dataset on this LeRobotDataset -- falling back to "
              "indexed access (decodes video; slower to load)")
    return hf


def episode_range(dataset, episode_idx):
    """[start, end) row indices of one episode.

    `episode_data_index` is the documented accessor; when it is absent the same
    ranges are recovered from the `episode_index` column, which every version
    writes."""
    edi = getattr(dataset, "episode_data_index", None)
    if edi is not None:
        n_eps = len(edi["from"])
        if episode_idx < 0 or episode_idx >= n_eps:
            raise ValueError(
                f"episode_idx {episode_idx} out of range (dataset has {n_eps})")
        return int(edi["from"][episode_idx]), int(edi["to"][episode_idx]), n_eps

    hf = frame_table(dataset)
    if hf is None:
        raise RuntimeError(
            "cannot determine episode boundaries: this LeRobotDataset exposes "
            "neither episode_data_index nor hf_dataset")
    eps = np.asarray(hf["episode_index"])
    present = np.unique(eps)
    if episode_idx not in present:
        raise ValueError(
            f"episode_idx {episode_idx} not in dataset (has {present.tolist()})")
    rows = np.nonzero(eps == episode_idx)[0]
    return int(rows[0]), int(rows[-1]) + 1, len(present)


def load_episode(dataset, mode, start, end):
    """(actions [N, D], timestamps [N]) for one episode, no video decoded."""
    arms = ARM_MODES[mode]
    ts_key = f"observation.timestamps.{arms[0]}"

    hf = frame_table(dataset)
    if hf is not None:
        rows = hf.select(range(start, end))
        ## lerobot may have put the table in torch format, in which case a
        ## column is a list of tensors rather than a list of lists -- go
        ## through to_numpy_1d row by row so either shape lands the same.
        actions = np.asarray([to_numpy_1d(a) for a in rows["action"]],
                             dtype=np.float64)
        if ts_key in rows.column_names:
            stamps = np.asarray([float(to_numpy_1d(t)[0]) for t in rows[ts_key]],
                                dtype=np.float64)
        else:
            # Pre-timestamp datasets, and any version that stores only
            # lerobot's own per-episode `timestamp` column.
            print(f"[replay] {ts_key} absent; pacing from lerobot's "
                  "`timestamp` column")
            stamps = np.asarray([float(np.asarray(t).reshape(-1)[0])
                                 for t in rows["timestamp"]], dtype=np.float64)
    else:
        actions, stamps = [], []
        for i in range(start, end):
            sample = dataset[i]
            actions.append(to_numpy_1d(sample["action"]))
            stamps.append(float(to_numpy_1d(sample[ts_key])[0]))
        actions = np.asarray(actions, dtype=np.float64)
        stamps = np.asarray(stamps, dtype=np.float64)

    return actions, stamps


def parse_action(action, mode):
    layout = ACTION_LAYOUTS[mode]
    return {key: action[idx] for key, idx in layout.items()}


def align_middle_waist(cmd, measured_waist):
    """Put the recorded middle-waist angle in the frame the servo booted into.

    The waist is a multiturn joint and `urdf_to_driver` recorded it relative to
    whatever 2pi-equivalent frame the driver happened to be in THAT session.
    A later session can boot a full turn away (encoder wrap; Homing_Offset is
    inert in extended-position mode), and re-sending the recorded number
    verbatim would then command a whole physical revolution and wind the
    cables.  Shift by whole turns to the equivalent nearest the current
    reading -- the physical pose is identical, the winding is not."""
    k = np.round((measured_waist - cmd[0]) / (2 * np.pi))
    if k == 0:
        return cmd, 0.0
    out = np.asarray(cmd, dtype=float).copy()
    out[0] = cmd[0] + 2 * np.pi * k
    return out, float(2 * np.pi * k)


def resolve_mode(args, meta, action_dim):
    """Decide the action layout, preferring the dataset's own record of it."""
    recorded = meta.get("mode")
    mode = args.mode or recorded

    if mode is None:
        raise SystemExit(
            "cannot tell which arms this dataset recorded: it has no meta.json "
            f"and no --mode was given. Its action width is {action_dim}; pass "
            f"the matching --mode from {sorted(ARM_MODES)}.")

    if mode not in ARM_MODES or mode not in ACTION_LAYOUTS:
        raise SystemExit(f"--mode must be one of "
                         f"{sorted(set(ARM_MODES) & set(ACTION_LAYOUTS))}, "
                         f"got '{mode}'")

    if args.mode and recorded and args.mode != recorded:
        print(f"[mode] dataset recorded mode '{recorded}', replaying as "
              f"'{args.mode}' because --mode said so.")

    ## The layout is the whole safety argument here: a wrong one sends one
    ## arm's joint angles to a different arm without any error.  The action
    ## width is the one fact on disk that can refute it, so it is checked.
    want = action_width(mode)
    if want != action_dim:
        raise SystemExit(
            f"mode '{mode}' describes a {want}-value action, but this dataset's "
            f"actions are {action_dim} values wide -- the layout does not "
            f"belong to this recording. Recorded mode: {recorded or 'unknown'}.")

    return mode


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode's commanded joint trajectory.")
    parser.add_argument(
        "--mode",
        choices=sorted(set(ARM_MODES) & set(ACTION_LAYOUTS)),
        default=None,
        help="Action layout to use. Default: whatever the dataset's meta.json "
             "recorded.",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument(
        "--base-root",
        type=str,
        default=str(DATASET_ROOT),
        help="Where task folders live. Defaults to data_col_config.DATASET_ROOT, "
             "the same constant data collection writes to.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name, e.g. screwdriver_insertion")
    parser.add_argument("--run", type=str, default="latest", help="Run folder name or 'latest'")
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the replay rate. Default: pace from the "
                             "recorded timestamps.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and validate the episode; move nothing.")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step instead of a ~1 Hz summary.")
    args = parser.parse_args()

    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
    else:
        if args.task is None:
            raise SystemExit("Either --dataset-root OR (--task and optionally --run) must be provided")

        base_root = Path(args.base_root).expanduser().resolve()
        task_dir = base_root / args.task

        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

        if args.run == "latest":
            run_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                raise FileNotFoundError(f"No runs found under task directory: {task_dir}")
            # Run folders are named YYYYmmdd_HHMMSS, so lexical order is
            # chronological order.
            dataset_root = sorted(run_dirs)[-1]
        else:
            dataset_root = task_dir / args.run

        dataset_root = dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    print(f"Using dataset_root={dataset_root}")

    meta = read_dataset_meta(dataset_root)
    if meta:
        print(f"[meta] task={meta.get('task')} mode={meta.get('mode')} "
              f"arms={meta.get('active_arms')} fps={meta.get('fps')} "
              f"action_dim={meta.get('action_dim')} recorded {meta.get('created')}")
    else:
        print("[meta] no meta.json in this dataset -- mode must come from --mode")

    dataset = LeRobotDataset(str(dataset_root), video_backend="pyav")

    start, end, n_eps = episode_range(dataset, args.episode_idx)
    start = min(start + args.start_offset, end)
    if end - start <= 0:
        raise SystemExit("No steps to replay (check --start-offset)")

    ## Probe the width from row 0 so the mode can be validated BEFORE any arm
    ## is created, let alone energised.
    hf = frame_table(dataset)
    if hf is not None:
        action_dim = len(np.asarray(hf.select(range(start, start + 1))["action"])[0])
    else:
        action_dim = len(to_numpy_1d(dataset[start]["action"]))

    mode = resolve_mode(args, meta, action_dim)
    arm_names = ARM_MODES[mode]
    print(f"[mode] {mode} -> arms {arm_names}, action width {action_dim}")

    actions, stamps = load_episode(dataset, mode, start, end)

    total_steps = len(actions)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
        actions, stamps = actions[:total_steps], stamps[:total_steps]

    ## Recorded timestamps are time.monotonic() values, so only DIFFERENCES
    ## mean anything -- and a monotonic clock does not survive a reboot, which
    ## is exactly when a stale absolute value would look plausible.  Rebase on
    ## the first sample and sanity-check the spacing.
    rel = stamps - stamps[0]
    if total_steps > 1:
        dts = np.diff(rel)
        bad = (dts <= 0) | (dts > 1.0)
        if bad.any():
            print(f"[timing] {int(bad.sum())} of {len(dts)} recorded intervals "
                  "are non-monotonic or over 1 s -- pacing from the dataset fps "
                  "instead")
            rel = np.arange(total_steps) / float(dataset.fps)
        else:
            print(f"[timing] recorded interval: median {np.median(dts) * 1e3:.1f} ms "
                  f"(~{1.0 / max(np.median(dts), 1e-6):.1f} Hz), "
                  f"max {dts.max() * 1e3:.1f} ms")
    if args.fps is not None:
        rel = np.arange(total_steps) / float(args.fps)
        print(f"[timing] --fps {args.fps} overrides the recorded pacing")

    parsed_first = parse_action(actions[0], mode)
    print(f"episode {args.episode_idx} of {n_eps}: rows [{start}, {end}), "
          f"replaying {total_steps} steps, "
          f"{rel[-1] if total_steps else 0:.1f} s")
    for arm in arm_names:
        key = f"{arm}_arm"
        if key in parsed_first:
            print(f"  first {arm} command: {np.round(parsed_first[key], 3)}")

    ## Every command is length-checked here, once, rather than per step in the
    ## timed loop where a bad row would only be discovered mid-motion.
    for arm in arm_names:
        n = ARM_CONFIG[arm]["num_joints"]
        got = len(parsed_first[f"{arm}_arm"])
        if got != n:
            raise SystemExit(
                f"{arm}: layout yields {got} joints, arm has {n}")
    if not np.isfinite(actions).all():
        n_bad = int((~np.isfinite(actions)).any(axis=1).sum())
        raise SystemExit(
            f"{n_bad} of {total_steps} recorded actions contain NaN/inf -- "
            "refusing to replay a trajectory with holes in it")

    if args.dry_run:
        print("Dry run: episode loads, layout matches, no NaNs. "
              "Exiting before robot motion.")
        return

    ## Checked HERE rather than at the top of main(): everything above this
    ## line is a dataset question, and --dry-run answering it off the robot
    ## (no ROS, no arms) is the point of having the flag.
    if rospy is None:
        raise ImportError("rospy is required to replay episodes on hardware "
                          "(--dry-run works without it).")

    rospy.init_node("replay_episode", anonymous=True)

    robots = {arm_name: create_and_configure_robot(arm_name)
              for arm_name in arm_names}

    stop_requested = False

    def request_stop():
        nonlocal stop_requested
        stop_requested = True
        stop_robots(robots)

    rospy.on_shutdown(request_stop)

    ## Waist frame alignment, decided once from the pose measured after reset
    ## and applied to every middle command, so the whole trajectory stays in
    ## one frame instead of each command being re-judged independently.
    waist_turn_shift = 0.0

    try:
        for arm_name, bot in robots.items():
            reset_arm(bot, arm_name)
            n = ARM_CONFIG[arm_name]["num_joints"]
            measured_q = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
            print(f"Measured {arm_name} joints after reset:", np.round(measured_q, 4))

            first_cmd = np.asarray(parsed_first[f"{arm_name}_arm"], dtype=float)
            if arm_name == "middle":
                first_cmd, waist_turn_shift = align_middle_waist(
                    first_cmd, measured_q[0])
                if waist_turn_shift:
                    print(f"[frame] middle waist recorded at "
                          f"{parsed_first['middle_arm'][0]:+.3f}; replaying at "
                          f"{first_cmd[0]:+.3f} (nearest 2pi-equivalent to the "
                          f"measured {measured_q[0]:+.3f})")

            ## Walk to the episode's first commanded pose at the same
            ## step-limited rate the startup poses use.  Sending it in one
            ## jump would fail the driver's velocity check, which rejects the
            ## whole group command rather than clipping it.
            gap = float(np.max(np.abs(first_cmd - measured_q)))
            print(f"[start] {arm_name}: {gap:.3f} rad from reset pose to the "
                  "episode's first command; interpolating.")
            interpolate_to_pose(bot, arm_name, first_cmd)

            ## interpolate_to_pose RETURNS WITHOUT MOVING when its wrapped-
            ## encoder guard fires (a multiturn joint reading outside the
            ## driver's own limits), and it says so on stdout rather than
            ## raising.  Streaming a whole episode at an arm that never left
            ## its reset pose is exactly the situation that guard exists to
            ## prevent, so confirm the arm is actually where the trajectory
            ## starts before sending anything at it.
            rospy.sleep(0.1)
            landed = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
            residual = float(np.max(np.abs(first_cmd - landed)))
            if residual > START_POSE_TOLERANCE:
                raise SystemExit(
                    f"{arm_name}: still {residual:.3f} rad from the episode's "
                    f"first command after interpolating (tolerance "
                    f"{START_POSE_TOLERANCE:.3f}). Look above for a [SAFETY] "
                    "or driver-limit message -- replaying from here would "
                    "stream the whole trajectory at an arm standing "
                    "somewhere else.")
            print(f"[start] {arm_name}: in position "
                  f"({residual:.3f} rad residual).")

        print(f"Starting replay of {total_steps} steps. Press Ctrl+C to stop.")

        t0_wall = time.monotonic()
        lag_max = 0.0
        lagging = 0
        err_sum = {arm: 0.0 for arm in arm_names}
        err_max = {arm: 0.0 for arm in arm_names}
        err_n = 0
        next_report = t0_wall + 1.0

        for i in range(total_steps):
            if stop_requested or rospy.is_shutdown():
                print(f"\nStopped at step {i} of {total_steps}.")
                break

            sleep_time = (t0_wall + rel[i]) - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                lag = -sleep_time
                if lag > 0.01:
                    lagging += 1
                    lag_max = max(lag_max, lag)

            parsed = parse_action(actions[i], mode)

            for arm_name, bot in robots.items():
                cmd = np.asarray(parsed[f"{arm_name}_arm"], dtype=float)
                if arm_name == "middle" and waist_turn_shift:
                    cmd = cmd.copy()
                    cmd[0] += waist_turn_shift
                replay_arm_command(bot, cmd)

                gripper_key = f"{arm_name}_gripper"
                if gripper_key in parsed:
                    command_gripper(bot, float(parsed[gripper_key]))

            ## Reporting reads the joint states the driver publishes
            ## asynchronously -- no extra wait, so it costs the loop nothing
            ## and the error it shows is one tick stale by construction.
            now = time.monotonic()
            if args.verbose or now >= next_report or i == total_steps - 1:
                for arm_name, bot in robots.items():
                    n = ARM_CONFIG[arm_name]["num_joints"]
                    measured_q = np.asarray(
                        bot.dxl.joint_states.position[:n], dtype=float)
                    cmd = np.asarray(parsed[f"{arm_name}_arm"], dtype=float)
                    if arm_name == "middle" and waist_turn_shift:
                        cmd = cmd.copy()
                        cmd[0] += waist_turn_shift
                    err = float(np.linalg.norm(measured_q - cmd))
                    err_sum[arm_name] += err
                    err_max[arm_name] = max(err_max[arm_name], err)
                    if args.verbose:
                        print(f"{arm_name} target={np.round(cmd, 3)}")
                        print(f"{arm_name} measured={np.round(measured_q, 3)}")
                    print(f"[{i + 1}/{total_steps}] {arm_name} "
                          f"tracking_error={err:.4f}")
                err_n += 1
                next_report = now + 1.0

        elapsed = time.monotonic() - t0_wall
        print(f"\nReplayed {i + 1}/{total_steps} steps in {elapsed:.1f} s "
              f"(recorded {rel[total_steps - 1]:.1f} s)")
        if lagging:
            print(f"[timing] {lagging} steps ran late, worst "
                  f"{lag_max * 1e3:.0f} ms behind the recording")
        if err_n:
            for arm_name in arm_names:
                print(f"[tracking] {arm_name}: mean "
                      f"{err_sum[arm_name] / err_n:.4f} rad, max "
                      f"{err_max[arm_name]:.4f} rad (sampled {err_n}x)")

    except KeyboardInterrupt:
        request_stop()
    finally:
        stop_robots(robots)


if __name__ == "__main__":
    main()
