import os
import threading
import cv2
import jaxlie
import numpy as np
import pyrealsense2 as rs
import rospy
import torch
import time
import logging
import sys
sys.path.append("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts")
sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki as pk
import pyroki_snippets as pks
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF
from lerobot.datasets import LeRobotDataset
from headset_link import make_headset
from transform_utils import pose2mat

logging.basicConfig(
    filename="teleop_timing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def now():
    return time.monotonic()

# Constants 

URDF_PATH = "/home/devi/giava/right.urdf"
RIGHT_EE_LINK = "right_gripper_base"

DATASET_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot"

CAMERA_SERIALS = {
    "right_wrist": "230322270105",
    "top_scene": "230322270396",
}

TASKS = {
    1: "screwdriver_insertion",
    2: "block_square",
    3: "grasp cube",
}

RIGHT_ARM_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]

RIGHT_START_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
RIGHT_REST_Q = np.array([0.0, -1.6, 1.5, 0.0, 0.7, 0.0], dtype=float)

GRIPPER_CURRENT_LIMIT = 200

latest_frames = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
latest_timestamps = {
    name: {"color": None, "depth": None} for name in CAMERA_SERIALS
}
frame_lock = threading.Lock()
camera_shutdown = False
latest_key = None

# Data classes

@dataclass
class TeleopConfig:
    control_dt: float = 1.0 / 50.0
    position_scale: float = 1.35
    alpha: float = 0.3
    arm_cmd_dt: float = 1.0 / 50.0
    moving_time: float = 0.14
    accel_time: float = 0.04
    max_ee_step: float = 0.02
    pos_weight: float = 40.0
    ori_weight: float = 0.25
    dq_weight: float = 0.18
    joint_reached_tol: float = 0.03
    ee_reached_tol: float = 0.01
    cmd_timeout: float = 0.25
    full_joint_velocity_limits_value: float = 2.3

    # Per-instance NumPy arrays via default_factory
    max_joint_step: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.05, 0.06, 0.08, 0.08, 0.10],
            dtype=float,
        )
    )
    R_remap_right: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=float,
        )
    )

@dataclass
class TeleopSessionState:
    active: bool = False
    right_start_controller_pos: Optional[np.ndarray] = None
    right_start_controller_rot: Optional[np.ndarray] = None
    right_start_robot_pos: Optional[np.ndarray] = None
    right_start_robot_rot: Optional[np.ndarray] = None
    right_filtered_target_pos: Optional[np.ndarray] = None


@dataclass
class RobotCommandState:
    last_arm_cmd_time: float = 0.0
    last_right_cmd: Optional[np.ndarray] = None


@dataclass
class CommandKinematicsState:
    q_cmd: Optional[np.ndarray] = None
    T_right_cmd: Optional[np.ndarray] = None

@dataclass
class TimingStats:
    loop: list[float] = field(default_factory=list)
    headset: list[float] = field(default_factory=list)
    ik_solve: list[float] = field(default_factory=list)
    cmd: list[float] = field(default_factory=list)
    overruns: int = 0


@dataclass
class EpisodeStats:
    frames_added: int = 0
    frames_skipped_wrist_camera_not_ready: int = 0
    frames_skipped_top_camera_not_ready: int = 0
    frames_skipped_wrist_timestamp_missing: int = 0
    frames_skipped_top_timestamp_missing: int = 0
    frames_skipped_wrist_depth_not_ready: int = 0
    frames_skipped_top_depth_not_ready: int = 0
    frames_skipped_wrist_depth_timestamp_missing: int = 0
    frames_skipped_top_depth_timestamp_missing: int = 0
    ik_failures: int = 0
    teleop_enable_count: int = 0
    teleop_disable_count: int = 0
    robot_wrist_dt: list[float] = field(default_factory=list)
    robot_top_dt: list[float] = field(default_factory=list)
    cmd_track_err: list[float] = field(default_factory=list)


# Helper functions

def set_register(robot_name, motor_name, reg_name, value):
    service_name = f"/{robot_name}/set_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)
    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name
    req.value = value
    return srv(req)

def digital_zoom(frame, zoom=1.6):
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def clamp_joint_step(q_curr, q_target, max_step):
    dq = np.clip(q_target - q_curr, -max_step, max_step)
    return q_curr + dq

def clamp_cartesian_step(target, prev_target, max_step):
    delta = target - prev_target
    norm = np.linalg.norm(delta)
    if norm > max_step and norm > 1e-9:
        delta *= (max_step / norm)
    return prev_target + delta

def quat_xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

def matrix_to_pose7(T):
    pos = T[:3, 3].astype(np.float32)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    return np.concatenate([pos, quat_wxyz], axis=0)

def start_teleop_session(state, right_controller, T_right):
    state.active = True
    state.right_start_controller_pos = right_controller[:3, 3].copy()
    state.right_start_controller_rot = right_controller[:3, :3].copy()
    state.right_start_robot_pos = T_right[:3, 3].copy()
    state.right_start_robot_rot = T_right[:3, :3].copy()
    state.right_filtered_target_pos = T_right[:3, 3].copy()

def compute_right_target(cfg, state, right_controller, T_right):
    right_delta_ctrl = right_controller[:3, 3] - state.right_start_controller_pos
    right_delta_robot = cfg.R_remap_right @ right_delta_ctrl

    right_raw = state.right_start_robot_pos + cfg.position_scale * right_delta_robot
    state.right_filtered_target_pos = (
        cfg.alpha * right_raw + (1.0 - cfg.alpha) * state.right_filtered_target_pos
    )

    right_target_pos = clamp_cartesian_step(
        state.right_filtered_target_pos,
        T_right[:3, 3].copy(),
        cfg.max_ee_step,
    )

    right_delta_rot = right_controller[:3, :3] @ state.right_start_controller_rot.T
    right_target_rot = right_delta_rot @ state.right_start_robot_rot

    right_target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(right_target_rot).as_quat())
    return right_target_pos, right_target_wxyz

def stop_teleop_session(state):
    state.active = False

def update_gripper(bot, trigger_pressed, close_position=-1.5, open_position=0.0):
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = close_position if trigger_pressed else open_position
    bot.gripper.core.pub_single.publish(cmd)
    return cmd.cmd

def safe_move_arm_joints(bot, target_q, total_time=3.0, step_time=0.25, accel_ratio=0.35):
    current_q = np.array(bot.dxl.joint_states.position[:6], dtype=float)
    target_q = np.array(target_q, dtype=float)
    max_step = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)

    delta = target_q - current_q
    n_steps = int(np.ceil(np.max(np.abs(delta) / max_step)))
    n_steps = max(n_steps, int(np.ceil(total_time / step_time)), 1)

    waypoints = np.linspace(current_q, target_q, n_steps + 1)[1:]
    move_t = step_time
    accel_t = min(accel_ratio * move_t, 0.5 * move_t)

    for q_cmd in waypoints:
        bot.arm.set_joint_positions(q_cmd.tolist(), moving_time=move_t, accel_time=accel_t, blocking=True)

def move_to_pose(
    right_bot,
    q,
    cmd_kin,
    cmd_state,
    right_arm_indices,
    robot,
    right_ee_index,
    cfg,
    full_joint_velocity_limits,
    target_q,
    description="pose",
    total_time=4.0,
    step_time=0.3,
):
    print(f"\nMoving RIGHT arm safely to {description}...")
    safe_move_arm_joints(right_bot, target_q, total_time=total_time, step_time=step_time)

    # Update internal state to match
    q[right_arm_indices] = target_q.copy()

    _, T_right = refresh_fk(robot, q, right_ee_index)
    cmd_kin.q_cmd = q.copy()
    cmd_kin.T_right_cmd = T_right.copy()
    cmd_state.last_right_cmd = target_q.copy()
    cmd_state.last_arm_cmd_time = 0.0

def move_to_start_pose(
    right_bot,
    q,
    cmd_kin,
    cmd_state,
    right_arm_indices,
    robot,
    right_ee_index,
    cfg,
    full_joint_velocity_limits,
):
    move_to_pose(
        right_bot,
        q,
        cmd_kin,
        cmd_state,
        right_arm_indices,
        robot,
        right_ee_index,
        cfg,
        full_joint_velocity_limits,
        target_q=RIGHT_START_Q,
        description="START pose",
    )

def move_to_rest_pose(
    right_bot,
    q,
    cmd_kin,
    cmd_state,
    right_arm_indices,
    robot,
    right_ee_index,
    cfg,
    full_joint_velocity_limits,
):
    move_to_pose(
        right_bot,
        q,
        cmd_kin,
        cmd_state,
        right_arm_indices,
        robot,
        right_ee_index,
        cfg,
        full_joint_velocity_limits,
        target_q=RIGHT_REST_Q,
        description="FINAL REST pose",
        total_time=4.0,
        step_time=0.3,
    )
def keyboard_listener():
    global latest_key
    while True:
        latest_key = input().strip()

def camera_worker(name, pipeline, align):
    global camera_shutdown

    while not camera_shutdown:
        try:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
        except Exception as e:
            print(f"Warning: Waiting for camera frames failed: {e}")
            continue

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())  # uint16 z16 depth image
        color_ts = color_frame.get_timestamp() * 1e-3
        depth_ts = depth_frame.get_timestamp() * 1e-3

        with frame_lock:
            latest_frames[name]["color"] = color
            latest_frames[name]["depth"] = depth
            latest_timestamps[name]["color"] = color_ts
            latest_timestamps[name]["depth"] = depth_ts

def reset_stats():
    return TimingStats(), EpisodeStats()

def build_robot_model():
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    right_arm_indices = [robot.joints.actuated_names.index(name) for name in RIGHT_ARM_NAMES]

    right_ee_index = robot.links.names.index(RIGHT_EE_LINK)

    return robot, right_arm_indices, right_ee_index

def create_bots(cfg):
    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=cfg.moving_time,
        accel_time=cfg.accel_time,
        init_node=False,
    )
    right_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_register("puppet_right", "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    return right_bot

def solve_single_arm_ik(
    robot,
    target_link_name,
    target_position,
    target_wxyz,
    prev_q,
    dt,
    joint_velocity_limits,
    pos_weight=40.0,
    ori_weight=0.25,
    dq_weight=0.18,
):
    return pks.solve_trajectories_ik(
        robot=robot,
        target_link_names=[target_link_name],
        target_positions=[target_position],
        target_wxyzs=[target_wxyz],
        prev_q=prev_q,
        dt=dt,
        joint_velocity_limits=joint_velocity_limits,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        dq_weight=dq_weight,
    )

def initialize_bot(right_bot):
    right_bot.arm.set_joint_positions(RIGHT_START_Q.tolist(), moving_time=2.0, accel_time=0.5, blocking=True)
    time.sleep(2)
    rospy.sleep(2.0)

def setup_cameras():
    pipelines = {}
    aligns = {}
    intrinsics = {}

    for name, serial in CAMERA_SERIALS.items():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        depth_intr = depth_profile.get_intrinsics()

        pipelines[name] = pipeline
        aligns[name] = align
        intrinsics[name] = {
            "color": color_intr,
            "depth": depth_intr,
            "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
        }

        threading.Thread(
            target=camera_worker,
            args=(name, pipeline, align),
            daemon=True,
        ).start()

    return pipelines, intrinsics

def refresh_fk(robot, q, right_ee_index):
    fk = robot.forward_kinematics(q)
    T_right = jaxlie.SE3(fk[right_ee_index]).as_matrix()
    return fk, T_right

def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    """Ask user a yes/no question and return True for yes, False for no."""
    while True:
        suffix = " [y/n] "
        if default is True:
            suffix = " [Y/n] "
        elif default is False:
            suffix = " [y/N] "
        ans = input(prompt + suffix).strip().lower()

        if ans == "" and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")

def safe_shutdown(
    right_bot,
    pipelines,
    dataset,
    collecting_episode,
    task_name,
    episode_idx,
    robot,
    q,
    cmd_kin,
    cmd_state,
    right_arm_indices,
    right_ee_index,
    cfg,
    full_joint_velocity_limits,
    episode_stats=None,
):
    global camera_shutdown

    # Ask about unsaved in-memory episode data
    if collecting_episode:
        has_unsaved_data = True
        if episode_stats is not None:
            has_unsaved_data = episode_stats.frames_added > 0

        if has_unsaved_data:
            save_before_exit = ask_yes_no(
                "\nThere is recorded data from the current episode that has not been saved. Save before shutdown?",
                default=True,
            )

            if save_before_exit:
                try:
                    dataset.save_episode()
                    print(f"\nEPISODE SAVED: episode_{episode_idx:04d}")
                    episode_idx += 1
                except Exception as e:
                    print(f"\nWarning: failed to save episode before shutdown: {e}")
                    discard_anyway = ask_yes_no(
                        "Save failed. Discard unsaved episode and continue shutdown?",
                        default=False,
                    )
                    if not discard_anyway:
                        print("\nShutdown canceled.")
                        return episode_idx, False
            else:
                discard_episode = ask_yes_no(
                    "Discard unsaved recorded data and continue shutdown?",
                    default=False,
                )
                if not discard_episode:
                    print("\nShutdown canceled.")
                    return episode_idx, False

    # CHANGED (lerobot v3 dataset format): flush buffered episode metadata and
    # write parquet footers, otherwise the dataset on disk cannot be loaded back.
    try:
        dataset.finalize()
    except Exception as e:
        print(f"\nWarning: failed to finalize dataset: {e}")

    # Move robot to rest pose
    try:
        print("\nMoving arm to rest pose before shutdown...")
        move_to_rest_pose(
            right_bot,
            q,
            cmd_kin,
            cmd_state,
            right_arm_indices,
            robot,
            right_ee_index,
            cfg,
            full_joint_velocity_limits,
        )
        print("\nRobot at rest pose.")
    except Exception as e:
        print(f"\nWarning: failed to move robot to rest pose: {e}")

    # Stop cameras
    camera_shutdown = True
    for pipeline in pipelines.values():
        try:
            pipeline.stop()
        except Exception as e:
            print(f"Error stopping camera pipeline: {e}")

    print("\nShutdown complete.")
    return episode_idx, True

def main():
    global latest_key
    collecting_episode = False
    timing_stats = TimingStats()
    episode_stats = EpisodeStats()

    cfg = TeleopConfig()

    # Init ROS
    rospy.init_node("bimanual_vr_teleop")

    # Print tasks and get selection BEFORE starting headset / cameras / keyboard
    print("\nAVAILABLE TASKS:\n")
    for idx, name in TASKS.items():
        print(f"{idx}: {name}")

    # robust selection
    while True:
        try:
            task_idx = int(input("\nSelect task number: ").strip())
            if task_idx not in TASKS:
                print("Invalid task number. Try again.")
                continue
            break
        except ValueError:
            print("Please enter an integer task number.")

    task_name = TASKS[task_idx]
    print(f"\nSelected task: {task_name}")

    # background keyboard thread to get user input
    threading.Thread(target=keyboard_listener, daemon=True).start()

    # headset thread
    headset = make_headset()
    headset.run_in_thread()

    # camera pipelines
    pipelines, intrinsics = setup_cameras()

    # robot initialization
    robot, right_arm_indices, right_ee_index = build_robot_model()
    right_bot = create_bots(cfg)
    initialize_bot(right_bot)
    update_gripper(right_bot, False)
    rospy.sleep(0.5)

    q = np.zeros(robot.joints.num_actuated_joints)
    q[right_arm_indices] = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)

    # calculate the pose of the end effector using fk
    fk, T_right = refresh_fk(robot, q, right_ee_index)
    cmd_kin = CommandKinematicsState(q_cmd=q.copy(), T_right_cmd=T_right.copy())
    full_joint_velocity_limits = np.ones(robot.joints.num_actuated_joints) * cfg.full_joint_velocity_limits_value

    teleop_state = TeleopSessionState()
    cmd_state = RobotCommandState(
        last_right_cmd=np.array(right_bot.dxl.joint_states.position[:6], dtype=float),
    )

    episode_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    right_gripper_action = 0.1

    repo_id = f"deviamar/{task_name}"
    run_name = time.strftime("%Y%m%d_%H%M%S")
    dataset_root = Path(DATASET_ROOT) / task_name / run_name
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_root),
        fps=round(1.0 / cfg.control_dt),
        features={
            "observation.images.right_wrist": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.top_scene": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "rightgripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "rightwaist_cmd",
                    "rightshoulder_cmd",
                    "rightelbow_cmd",
                    "rightforearm_roll_cmd",
                    "rightwrist_angle_cmd",
                    "rightwrist_rotate_cmd",
                    "rightgripper_cmd",
                ],
            },
            "observation.ee_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            },
            "observation.timestamps.robot": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "observation.timestamps.right_wrist": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "observation.timestamps.top_scene": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "observation.depth.right_wrist": {
                "dtype": "float32",
                "shape": (480, 640),
                "names": ["height", "width"],
            },
            "observation.depth.top_scene": {
                "dtype": "float32",
                "shape": (480, 640),
                "names": ["height", "width"],
            },
            "observation.depth_intrinsics.right_wrist": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["fx", "fy", "cx", "cy"],
            },
            "observation.depth_intrinsics.top_scene": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["fx", "fy", "cx", "cy"],
            },
            "observation.timestamps.right_wrist_depth": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "observation.timestamps.top_scene_depth": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    next_tick = now()

    while not rospy.is_shutdown():
        loop_start = now()

        key = latest_key
        latest_key = None

        if key == "i":
            stop_teleop_session(teleop_state)

            if collecting_episode:
                print("\nDiscarding current in-memory episode before resetting to start pose.")
                collecting_episode = False

            print("\nTeleop DISABLED. MOVING TO START POSE")
            move_to_start_pose(
                right_bot,
                q,
                cmd_kin,
                cmd_state,
                right_arm_indices,
                robot,
                right_ee_index,
                cfg,
                full_joint_velocity_limits,
            )
            print("\nRobot ready to start.")
            continue

        if key == "s":
            print("Starting the save")
            try:
                if not collecting_episode:
                    print("\nNo active episode to save.")
                else:
                    collecting_episode = False
                    dataset.save_episode()
                    print("\nEPISODE SAVED")

                    logging.info(
                        "timing_summary episode=%04d mean_loop_ms=%.2f p95_loop_ms=%.2f "
                        "mean_ik_ms=%.2f mean_cmd_ms=%.2f overruns=%d",
                        episode_idx,
                        1000 * np.mean(timing_stats.loop) if timing_stats.loop else 0.0,
                        1000 * np.percentile(timing_stats.loop, 95) if timing_stats.loop else 0.0,
                        1000 * np.mean(timing_stats.ik_solve) if timing_stats.ik_solve else 0.0,
                        1000 * np.mean(timing_stats.cmd) if timing_stats.cmd else 0.0,
                        timing_stats.overruns,
                    )

                    logging.info(
                        "episode_summary episode=%04d frames_added=%d "
                        "skip_wrist_cam=%d skip_top_cam=%d "
                        "skip_wrist_ts=%d skip_top_ts=%d "
                        "ik_failures=%d mean_wrist_lag_ms=%.2f "
                        "mean_top_lag_ms=%.2f mean_cmd_track_err=%.5f max_cmd_track_err=%.5f",
                        episode_idx,
                        episode_stats.frames_added,
                        episode_stats.frames_skipped_wrist_camera_not_ready,
                        episode_stats.frames_skipped_top_camera_not_ready,
                        episode_stats.frames_skipped_wrist_timestamp_missing,
                        episode_stats.frames_skipped_top_timestamp_missing,
                        episode_stats.ik_failures,
                        1000 * np.mean(episode_stats.robot_wrist_dt) if episode_stats.robot_wrist_dt else 0.0,
                        1000 * np.mean(episode_stats.robot_top_dt) if episode_stats.robot_top_dt else 0.0,
                        np.mean(episode_stats.cmd_track_err) if episode_stats.cmd_track_err else 0.0,
                        np.max(episode_stats.cmd_track_err) if episode_stats.cmd_track_err else 0.0,
                    )
                    print("\nLOGGING COMPLETE")
                    episode_idx += 1

            except Exception as e:
                print(f"\nEpisode save failed: {e}")

        # removing "d" as data is already being saved 
        # if key == "d":
        #     if collecting_episode:
        #         collecting_episode = False
        #         reset_timing_log()
        #         print("\nEPISODE DISCARDED")

        
        # ====== START NEW EPISODE ("r") ======
        if key == "r" and not collecting_episode:
            print("\nCOLLECTING NEW EPISODE")

            collecting_episode = True

            timing_stats = TimingStats()
            episode_stats = EpisodeStats()

            print(f"\nCOLLECTING: episode_{episode_idx:04d}")
        
        if key == "q":
            print("\nExiting...")

            episode_idx, did_shutdown = safe_shutdown(
                right_bot=right_bot,
                pipelines=pipelines,
                dataset=dataset,
                collecting_episode=collecting_episode,
                task_name=task_name,
                episode_idx=episode_idx,
                robot=robot,
                q=q,
                cmd_kin=cmd_kin,
                cmd_state=cmd_state,
                right_arm_indices=right_arm_indices,
                right_ee_index=right_ee_index,
                cfg=cfg,
                full_joint_velocity_limits=full_joint_velocity_limits,
                episode_stats=episode_stats,
            )

            if did_shutdown:
                break

        # --- headset, teleop, IK code here ---


        # TELEOPERATING

        t_headset_start = now()
        headset_data = headset.receive_data()
        timing_stats.headset.append(now() - t_headset_start)

        if headset_data is None:
            continue

        right_controller = pose2mat(headset_data.r_pos, headset_data.r_quat)
        button_pressed = headset_data.r_button_one
        right_trigger = headset_data.r_index_trigger

        if button_pressed and not teleop_state.active:
            start_teleop_session(teleop_state, right_controller, cmd_kin.T_right_cmd)
            print("\nTeleop ENABLED")
            episode_stats.teleop_enable_count += 1
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")
            episode_stats.teleop_disable_count += 1

        right_gripper_action = update_gripper(right_bot, right_trigger > 0)
        right_joint_state_msg = right_bot.dxl.joint_states
        right_joint_positions = np.array(right_joint_state_msg.position[:6], dtype=np.float32)
        right_gripper_state = np.array([right_joint_state_msg.position[6]], dtype=np.float32)

        if teleop_state.active:
            ik_attempted = False
            q_new = None
            t_solve_start = now()

            right_target_pos, right_target_wxyz = compute_right_target(
                cfg,
                teleop_state,
                right_controller,
                cmd_kin.T_right_cmd,
            )

            if (t_solve_start - cmd_state.last_arm_cmd_time) >= cfg.arm_cmd_dt:
                ik_attempted = True
                q_new = solve_single_arm_ik(
                    robot=robot,
                    target_link_name=RIGHT_EE_LINK,
                    target_position=right_target_pos,
                    target_wxyz=right_target_wxyz,
                    prev_q=cmd_kin.q_cmd,
                    dt=cfg.arm_cmd_dt,
                    joint_velocity_limits=full_joint_velocity_limits,
                    pos_weight=cfg.pos_weight,
                    ori_weight=cfg.ori_weight,
                    dq_weight=cfg.dq_weight,
                )

                if q_new is not None:
                    right_target_q = np.asarray(q_new[right_arm_indices], dtype=float)
                    right_prev_cmd = (
                        cmd_state.last_right_cmd
                        if cmd_state.last_right_cmd is not None
                        else cmd_kin.q_cmd[right_arm_indices]
                    )

                    right_q_cmd = clamp_joint_step(right_prev_cmd, right_target_q, cfg.max_joint_step)

                    t_cmd_start = now()

                    right_bot.arm.set_joint_positions(
                        right_q_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )

                    timing_stats.cmd.append(now() - t_cmd_start)

                    cmd_kin.q_cmd[right_arm_indices] = right_q_cmd

                    fk_cmd = robot.forward_kinematics(cmd_kin.q_cmd)
                    cmd_kin.T_right_cmd = jaxlie.SE3(fk_cmd[right_ee_index]).as_matrix()

                    cmd_state.last_right_cmd = right_q_cmd.copy()
                    cmd_state.last_arm_cmd_time = now()

            timing_stats.ik_solve.append(now() - t_solve_start)

            if ik_attempted and q_new is None:
                episode_stats.ik_failures += 1

        if collecting_episode:
            obs_ts = now()

            # =========================
            # CAMERA SNAPSHOT
            # =========================
            
            with frame_lock:
                wrist_color_raw = latest_frames["right_wrist"]["color"]
                wrist_depth_raw = latest_frames["right_wrist"]["depth"]
                top_color_raw = latest_frames["top_scene"]["color"]
                top_depth_raw = latest_frames["top_scene"]["depth"]

                wrist_color_ts = latest_timestamps["right_wrist"]["color"]
                wrist_depth_ts = latest_timestamps["right_wrist"]["depth"]
                top_color_ts = latest_timestamps["top_scene"]["color"]
                top_depth_ts = latest_timestamps["top_scene"]["depth"]

            wrist_frame = None if wrist_color_raw is None else wrist_color_raw.copy()
            top_frame = None if top_color_raw is None else top_color_raw.copy()

            wrist_depth = None if wrist_depth_raw is None else (
                wrist_depth_raw.astype(np.float32) * intrinsics["right_wrist"]["depth_scale"]
            )
            top_depth = None if top_depth_raw is None else (
                top_depth_raw.astype(np.float32) * intrinsics["top_scene"]["depth_scale"]
            )

            wrist_frame_missing = wrist_frame is None
            top_frame_missing = top_frame is None
            wrist_depth_missing = wrist_depth is None
            top_depth_missing = top_depth is None
            wrist_ts_missing = wrist_color_ts is None
            top_ts_missing = top_color_ts is None
            wrist_depth_ts_missing = wrist_depth_ts is None
            top_depth_ts_missing = top_depth_ts is None

            if wrist_frame_missing:
                episode_stats.frames_skipped_wrist_camera_not_ready += 1

            if top_frame_missing:
                episode_stats.frames_skipped_top_camera_not_ready += 1

            if wrist_ts_missing:
                episode_stats.frames_skipped_wrist_timestamp_missing += 1

            if top_ts_missing:
                episode_stats.frames_skipped_top_timestamp_missing += 1
            
            if wrist_depth_missing:
                episode_stats.frames_skipped_wrist_depth_not_ready += 1
            if top_depth_missing:
                episode_stats.frames_skipped_top_depth_not_ready += 1
            if wrist_depth_ts_missing:
                episode_stats.frames_skipped_wrist_depth_timestamp_missing += 1
            if top_depth_ts_missing:
                episode_stats.frames_skipped_top_depth_timestamp_missing += 1

            if wrist_frame_missing or top_frame_missing:
                continue

            if wrist_ts_missing or top_ts_missing:
                continue

            if wrist_depth_missing or top_depth_missing:
                continue

            if wrist_depth_ts_missing or top_depth_ts_missing:
                continue

            top_frame = digital_zoom(top_frame)

            episode_stats.frames_added += 1
            episode_stats.robot_wrist_dt.append(obs_ts - wrist_color_ts)
            episode_stats.robot_top_dt.append(obs_ts - top_color_ts)
            
            

            # =========================
            # ACTION
            # =========================

            commanded_joint_positions = (
                right_joint_positions.copy()
                if cmd_state.last_right_cmd is None
                else np.asarray(
                    cmd_state.last_right_cmd,
                    dtype=np.float32,
                ).copy()
            )

            commanded_gripper = np.array(
                [right_gripper_action],
                dtype=np.float32,
            )

            episode_stats.cmd_track_err.append(
                float(np.linalg.norm(commanded_joint_positions - right_joint_positions))
            )

            # =========================
            # FK / EE POSE
            # =========================

            q_measured = np.zeros(
                robot.joints.num_actuated_joints,
                dtype=float,
            )

            q_measured[right_arm_indices] = (
                right_joint_positions.astype(float)
            )

            _, T_right_measured = refresh_fk(
                robot,
                q_measured,
                right_ee_index,
            )

            ee_pose = matrix_to_pose7(
                T_right_measured
            )

            # =========================
            # BUILD FRAME
            # =========================

            observation_state = np.concatenate(
                [right_joint_positions, right_gripper_state],
                axis=0,
            )

            action = np.concatenate(
                [commanded_joint_positions, commanded_gripper],
                axis=0,
            )

            frame = {
                "observation.images.right_wrist": torch.from_numpy(wrist_frame),
                "observation.images.top_scene": torch.from_numpy(top_frame),
                "observation.depth.right_wrist": torch.from_numpy(wrist_depth),
                "observation.depth.top_scene": torch.from_numpy(top_depth),

                "observation.state": torch.from_numpy(observation_state),
                "action": torch.from_numpy(action),
                "observation.ee_pose": torch.from_numpy(ee_pose),

                "observation.timestamps.robot": torch.tensor([obs_ts], dtype=torch.float32),
                "observation.timestamps.right_wrist": torch.tensor([wrist_color_ts], dtype=torch.float32),
                "observation.timestamps.top_scene": torch.tensor([top_color_ts], dtype=torch.float32),
                "observation.timestamps.right_wrist_depth": torch.tensor([wrist_depth_ts], dtype=torch.float32),
                "observation.timestamps.top_scene_depth": torch.tensor([top_depth_ts], dtype=torch.float32),

                "observation.depth_intrinsics.right_wrist": torch.tensor([
                    intrinsics["right_wrist"]["color"].fx,
                    intrinsics["right_wrist"]["color"].fy,
                    intrinsics["right_wrist"]["color"].ppx,
                    intrinsics["right_wrist"]["color"].ppy,
                ], dtype=torch.float32),

                "observation.depth_intrinsics.top_scene": torch.tensor([
                    intrinsics["top_scene"]["color"].fx,
                    intrinsics["top_scene"]["color"].fy,
                    intrinsics["top_scene"]["color"].ppx,
                    intrinsics["top_scene"]["color"].ppy,
                ], dtype=torch.float32),
            }

            dataset.add_frame(frame, task_name)
        
        loop_elapsed = now() - loop_start
        timing_stats.loop.append(loop_elapsed)
        if loop_elapsed > cfg.control_dt:
            timing_stats.overruns += 1

        # improved sleep to reduce drift
        next_tick += cfg.control_dt

        sleep_time = next_tick - now()

        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()