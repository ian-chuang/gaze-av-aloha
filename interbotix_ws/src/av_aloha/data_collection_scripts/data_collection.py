import sys
import time
import threading

import numpy as np
import rospy
import torch
import jaxlie

from webrtc_headset import WebRTCHeadset

from arm_config import ARM_CONFIG

from camera_manager import (
    CameraConfig,
    CAMERA_SERIALS,
    setup_cameras,
    get_active_cameras,
    digital_zoom,
)

from robot_control import (
    create_and_configure_robots,
    build_robot_model,
    move_to_named_pose,
    reset_arms,
    compute_fk_and_ee,
)

from gripper import update_gripper

from dataset import (
    create_dataset,
    build_frame,
)

from data_col_config import (
    TASKS,
    ARM_MODES,
    ArmTeleopState,
    TeleopConfig,
    TeleopSessionState,
    RobotCommandState,
    CommandKinematicsState,
    compute_camera_arm_target,
    compute_gripper_arm_target,
    start_teleop_session,
    stop_teleop_session,
    solve_single_arm_ik,
    clamp_joint_step,
)

from transform_utils import pose2mat, quat2mat

from log import (
    SessionStats,
    reset_episode_log,
    log_episode_info,
)

latest_frames = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
latest_timestamps = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
frame_lock = threading.Lock()
camera_shutdown = threading.Event()
latest_key = None

camera_config = CameraConfig(
    top_active=True,
    low_active=True,
)

def keyboard_listener():
    global latest_key
    while True:
        latest_key = input().strip()

def select_mode():
    mode_names = list(ARM_MODES.keys())

    print("\nAVAILABLE MODES\n")

    for mode, arm_names in enumerate(mode_names):
        print(f"{mode}: {arm_names}")

    while True:
        try:
            idx = int(input("\nSelect mode: "))
            return mode_names[idx]
        except:
            print("Invalid selection")

def now():
    return time.monotonic()

def mark_episode_discarded(dataset_root, episode_idx):
    discard_file = dataset_root / "discarded_episodes.txt"
    with open(discard_file, "a") as f:
        f.write(f"{episode_idx}\n")

def safe_shutdown(robots, pipelines, dataset, collecting_episode, dataset_root, episode_idx):
    """Minimal safe shutdown helper to avoid undefined-symbol crashes."""
    try:
        if collecting_episode and dataset is not None:
            # try to save current episode
            dataset.save_episode()
    except Exception:
        pass

    try:
        # try to reset all robots to rest
        reset_arms(robots)
    except Exception:
        pass

    # close/cleanup pipelines if they expose stop/close (best-effort)
    try:
        for p in pipelines.values():
            stop = getattr(p, "stop", None)
            if callable(stop):
                stop()
    except Exception:
        pass

    return episode_idx, True

def main():

    rospy.init_node("data_collection")

    global latest_key
    collecting_episode = False
    cfg = TeleopConfig()

    stats = SessionStats()

    # # Print tasks and get selection
    # print("\nAVAILABLE TASKS:\n")
    # for idx, name in TASKS.items():
    #     print(f"{idx}: {name}")

    # while True:
    #     try:
    #         task_idx = int(input("\nSelect task number: ").strip())
    #         if task_idx not in TASKS:
    #             print("Invalid task number. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter an integer task number.")

    task_name = TASKS[5] # TASKS[task_idx]. Debugging bimanual data collection. Hardcoded for now.
    print(f"\nSelected task: {task_name}")

    # # Print modes and get selection
    # print("\nAVAILABLE MODES:\n")
    # for mode, arm_names in ARM_MODES.items():
    #     print(f"{mode}: {arm_names}")

    # while True:
    #     try:
    #         mode = input("\nSelect mode: ").strip()
    #         if mode not in ARM_MODES:
    #             print("Invalid mode. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter one of the listed modes.")

    mode = "bimanual" # hardcoded for now

    # # Print modes and get selection
    # print("\nWhich scene cameras to activate?\nEnter l for low and t for top (e.g. 'lt' for both, 'l' for low only, 't' for top only):")

    # while True:
    #     try:
    #         camera_selection = input().strip()
    #         if not camera_selection:
    #             print("Invalid selection. Try again.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Please enter a valid selection.")

    # camera_config = CameraConfig(
    #     top_active="t" in camera_selection,
    #     low_active="l" in camera_selection,
    # )

    camera_config = CameraConfig(
        top_active=True,
        low_active=True,
    )

    # background keyboard thread to get user input
    threading.Thread(target=keyboard_listener, daemon=True).start()

    # headset thread
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # camera pipelines
    # determine active cameras (use camera manager helper if present; fallback to configured serials)
    active_cameras = get_active_cameras(mode, camera_config)

    print(f"active cam: {active_cameras}")

    # initialize latest frame/timestamp containers now that active_cameras is known
    latest_frames = {cam: None for cam in active_cameras}
    latest_timestamps = {cam: None for cam in active_cameras}

    # setup cameras
    pipelines = setup_cameras(
        active_cameras,
        camera_shutdown,
        frame_lock,
        latest_frames,
        latest_timestamps,
    )

    arm_names = ARM_MODES[mode]

    episode_stats = reset_episode_log(active_cameras, arm_names)

    robots = create_and_configure_robots(arm_names)

    robot, arm_data = build_robot_model(mode)

    for arm_name in arm_names:
        move_to_named_pose(robots[arm_name], arm_name, "forward")

    q = np.zeros(robot.joints.num_actuated_joints, dtype=float)

    for arm_name in arm_names:
        q[arm_data[arm_name]["joint_indices"]] = np.asarray(
            robots[arm_name].dxl.joint_states.position[:6],
            dtype=float,
        )

    fk, ee = compute_fk_and_ee(robot, q, arm_data)

    # for arm in arm_names:
    #     print(
    #         arm,
    #         type(ee[arm]),
    #         np.shape(ee[arm]),
    #     )

    cmd_kin = CommandKinematicsState(q_cmd=q.copy(),
        T_cmd={
            arm: ee[arm].copy() for arm in arm_names
        },
    )

    # print(type(cmd_kin.T_cmd["left"]))
    # print(np.shape(cmd_kin.T_cmd["left"]))
    # print(cmd_kin.T_cmd["left"])

    # cmd_state = RobotCommandState(
    #     last_cmds={
    #         arm: q[arm_data[arm]["joint_indices"]].copy() for arm in arm_names
    #     }
    # )

    cmd_state = RobotCommandState(
        last_arm_cmd_time={arm: 0.0 for arm in arm_names},
        last_cmds={
            arm: q[arm_data[arm]["joint_indices"]].copy()
            for arm in arm_names
        },
    )

    teleop_state = TeleopSessionState()

    full_joint_velocity_limits = np.ones(robot.joints.num_actuated_joints) * cfg.full_joint_velocity_limits_value

    episode_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gripper_actions = {arm: 0.1 for arm in ARM_MODES[mode]}
    
    dataset, dataset_root = create_dataset(task_name, mode, active_cameras, cfg.control_dt)

    # print(f"Dataset: {dataset}")
    # print(f"Dataset root: {dataset_root}")

    next_tick = now()

    print("READY!")

    while not rospy.is_shutdown():
        loop_start = now()

        key = latest_key
        latest_key = None

        if key == "i":
            stop_teleop_session(teleop_state)

            if collecting_episode:
                print("\nIn the middle of collecting episode. Press s to save and stop.")
                continue

            for arm in arm_names:
                move_to_named_pose(robots[arm], arm, "forward")
            print("\nREADY")
            continue

        if key == "s":
            if not collecting_episode:
                print("\nNo active episode to save.")
            else:
                print("\nStarting the save")
                stop_teleop_session(teleop_state)
                collecting_episode = False
                dataset.save_episode()
                log_episode_info(episode_idx, episode_stats)
                episode_idx += 1
                print(f"\nEPISODE SAVED")

                # LOGGING TBD
            continue

        if key == "d":
            if not collecting_episode:
                print("\nNo active episode to discard.")
            else:
                print("\nDiscarding the current episode.")
                stop_teleop_session(teleop_state)
                collecting_episode = False
                dataset.discard_episode()
                episode_idx += 1
                episode_stats = None
                reset_episode_log(active_cameras, arm_names)
                print(f"\nEPISODE DISCARDED")
            continue

        if key == "r":
            if collecting_episode:
                print("\nAlready recording episode. Press s to save or d to discard.")
            else:
                collecting_episode = True

                episode_stats = reset_episode_log(active_cameras, arm_names)

                print(f"\nCOLLECTING: episode_{episode_idx:04d}")

        if key == "q":
            print("\nEXITING")

            episode_idx, did_shutdown = safe_shutdown(
                robots, pipelines, dataset, collecting_episode, dataset_root, episode_idx)

            if did_shutdown:
                break

        # Teleoperating

        t_headset_start = now()
        headset_data = headset.receive_data()
        episode_stats.headset.append(now() - t_headset_start)

        if headset_data is None:
            continue

        controller_poses = {
            "right": pose2mat(headset_data.r_pos, headset_data.r_quat),
            "left": pose2mat(headset_data.l_pos, headset_data.l_quat),
            "middle": pose2mat(headset_data.h_pos, headset_data.h_quat),
        }

        button_pressed = headset_data.r_button_one or headset_data.l_button_one

        arm_active = {
            "left": headset_data.l_button_one,
            "right": headset_data.r_button_one,
            "middle": (headset_data.l_button_one or headset_data.r_button_one),
        }

        for arm in arm_names:
            if arm not in teleop_state.arms:
                teleop_state.arms[arm] = ArmTeleopState()

            teleop_state.arms[arm].active = arm_active[arm]

        # teleop_state.active = any(
        #     arm_state.active
        #     for arm_state in teleop_state.arms.values()
        # )

        if button_pressed and not teleop_state.active:
            start_teleop_session(teleop_state, mode, controller_poses, cmd_kin)
            print("\nTeleop ENABLED")
            episode_stats.teleop_enable_count += 1
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")
            episode_stats.teleop_disable_count += 1

        gripper_actions = {}
        joint_positions = {}
        gripper_states = {}

        for arm in arm_names:
            joint_state_msg = robots[arm].dxl.joint_states

            joint_positions[arm] = np.asarray(joint_state_msg.position[:6], dtype=np.float32)

            if len(joint_state_msg.position) > 6:
                gripper_states[arm] = np.asarray([joint_state_msg.position[6]], dtype=np.float32)

        if "left" in arm_names:
            gripper_actions["left"] = update_gripper(robots["left"], headset_data.l_index_trigger > 0)

        if "right" in arm_names:
            gripper_actions["right"] = update_gripper(robots["right"], headset_data.r_index_trigger > 0)

        gripper_actions["middle"] = None

        targets = {}

        for arm in arm_names:

            arm_state = teleop_state.arms[arm]

            if not arm_state.active:
                continue

            controller_pose = controller_poses[arm]

            if arm == "middle":
                # print("target_pos", target_pos)
                # print("target_wxyz", target_wxyz)
                # print("middle active:", teleop_state.arms["middle"].active)
                # print("camera target")
                # print(target_pos)
                target_pos, target_wxyz = compute_camera_arm_target(
                    cfg,
                    arm_state,
                    controller_pose,
                    cmd_kin.T_cmd[arm],
                    cfg.R_cam_remap,
                )
            else:
                target_pos, target_wxyz = compute_gripper_arm_target(
                    cfg,
                    arm_state,
                    controller_pose,
                    cmd_kin.T_cmd[arm],
                    cfg.R_arm_remap,
                )

            targets[arm] = (target_pos, target_wxyz)

        if teleop_state.active:
            # start_teleop_session(teleop_state, mode, controller_poses, cmd_kin)
            ik_attempted = False
            q_new = None
            t_solve_start = now()

            for arm in arm_names:

                if not teleop_state.arms[arm].active:
                    continue

                if (t_solve_start - cmd_state.last_arm_cmd_time[arm]) < cfg.arm_cmd_dt:
                    continue

                target_pos, target_wxyz = targets[arm]

                episode_stats.arms[arm].ik_attempts += 1

                q_new = solve_single_arm_ik(
                    robot=robot,
                    target_link_name=ARM_CONFIG[arm]["ee_link"],
                    target_position=target_pos,
                    target_wxyz=target_wxyz,
                    prev_q=cmd_kin.q_cmd,
                    dt=cfg.arm_cmd_dt,
                    joint_velocity_limits=full_joint_velocity_limits,
                    pos_weight=cfg.pos_weight,
                    ori_weight=cfg.ori_weight,
                    dq_weight=cfg.dq_weight,
                )

                if q_new is None:
                    episode_stats.arms[arm].ik_failures += 1
                    continue

                episode_stats.arms[arm].ik_successes += 1

                fk_sol, ee_sol = compute_fk_and_ee(robot, q_new, arm_data)
                pos_err = np.linalg.norm(ee_sol[arm][4:] - target_pos)
                episode_stats.arms[arm].ik_position_errors.append(pos_err)
                
                quat = np.asarray(ee_sol[arm][:4], dtype=np.float64).copy()

                print("quat")
                print(type(quat), quat.dtype, quat.flags.writeable)

                print("target")
                print(type(target_wxyz))
                print(target_wxyz.dtype if hasattr(target_wxyz, "dtype") else "no dtype")
                print(target_wxyz.flags.writeable if hasattr(target_wxyz, "flags") else "no flags")

                R_sol = quat2mat(quat)
                print("R_sol OK")

                R_target = quat2mat(target_wxyz)
                print("R_target OK")

                # quat = np.asarray(ee_sol[arm][:4], dtype=np.float64).copy()
                # R_sol = quat2mat(quat)
                # R_target = quat2mat(target_wxyz)

                print(quat.dtype)
                print(quat.flags)

                print(type(target_wxyz))
                print(target_wxyz.dtype)
                print(target_wxyz.flags)

                R_err = R_sol.T @ R_target
                trace = np.trace(R_err)
                trace = np.clip(trace, -1.0, 3.0)
                angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
                episode_stats.arms[arm].ik_orientation_errors.append(angle_rad)

                joint_idx = arm_data[arm]["joint_indices"]

                if arm == "middle":
                    print("q_new", q_new[joint_idx])

                target_q = np.asarray(q_new[joint_idx], dtype=float)

                prev_cmd = (cmd_state.last_cmds[arm] if arm in cmd_state.last_cmds else cmd_kin.q_cmd[joint_idx])

                q_arm_cmd = clamp_joint_step(prev_cmd, target_q, cfg.max_joint_step)

                cmd_step = np.linalg.norm(q_arm_cmd - prev_cmd)

                episode_stats.arms[arm].joint_step_norms.append(cmd_step)



                t_cmd_start = now()

                robots[arm].arm.set_joint_positions(
                    q_arm_cmd.tolist(),
                    moving_time=cfg.moving_time,
                    accel_time=cfg.accel_time,
                    blocking=False,
                )

                if arm == "middle":
                    print("q_arm_cmd", q_arm_cmd)

                episode_stats.cmd.append(now() - t_cmd_start)

                cmd_kin.q_cmd[joint_idx] = q_arm_cmd

                cmd_state.last_cmds[arm] = q_arm_cmd.copy()
                cmd_state.last_arm_cmd_time[arm] = now()

            fk_cmd, ee_cmd = compute_fk_and_ee(robot, cmd_kin.q_cmd, arm_data)

            for arm in arm_names:
                cmd_kin.T_cmd[arm] = ee_cmd[arm]

            episode_stats.ik_solve.append(now() - t_solve_start)

        if collecting_episode:
            obs_ts = now()

            images = {}
            camera_timestamps = {}

            

            # Camera frames
            with frame_lock:
                for camera in active_cameras:

                    # if camera.startswith("oak"):
                    #     continue

                    frame = latest_frames.get(camera)
                    ts = latest_timestamps.get(camera)

                    if frame is None or ts is None:
                        episode_stats.cameras[camera].frames_missing += 1
                        continue

                    images[camera] = frame.copy()
                    camera_timestamps[camera] = ts


            # Robot states
            robot_states = {}

            for arm in arm_names:

                joints = joint_positions[arm].astype(np.float32)
                cmd_track_err = np.linalg.norm(joint_positions[arm] - cmd_state.last_cmds[arm])
                episode_stats.arms[arm].cmd_track_err.append(cmd_track_err)

                robot_states[arm] = {
                    "joints": joints,
                    "gripper": float(gripper_states[arm][0])
                    if arm in gripper_states
                    else 0.0,
                }

            robot_actions = {}

            for arm in arm_names:

                joint_idx = arm_data[arm]["joint_indices"]

                q_cmd = (
                    cmd_state.last_cmds[arm]
                    if arm in cmd_state.last_cmds
                    else cmd_kin.q_cmd[joint_idx]
                )

                robot_actions[arm] = {
                    "joints": np.asarray(q_cmd, dtype=np.float32),
                    "gripper": (gripper_actions[arm] if arm in gripper_actions else 0.0),
                }

            ee_poses = {}

            fk, ee = compute_fk_and_ee(
                robot,
                cmd_kin.q_cmd,
                arm_data,
            )

            for arm in arm_names:
                ee_poses[arm] = np.asarray(ee[arm], dtype=np.float32)

            timestamps = {
                **camera_timestamps,
            }

            for arm in arm_names:
                timestamps[arm] = obs_ts

            frame = build_frame(
                mode=mode,
                active_cameras=active_cameras,
                robot_states=robot_states,
                robot_actions=robot_actions,
                ee_poses=ee_poses,
                timestamps=timestamps,
                images=images,
            )

            dataset.add_frame(frame, task_name)

            episode_stats.frames_added += 1

        # sleep to reduce drift
        next_tick += cfg.control_dt
        sleep_time = next_tick - now()
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()