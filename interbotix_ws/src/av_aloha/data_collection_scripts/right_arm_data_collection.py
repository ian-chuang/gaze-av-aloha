import os
import sys
import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import jaxlie
import numpy as np
import pyroki as pk
import pyrealsense2 as rs
import rospy
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

sys.path.append("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts")
sys.path.append("/home/devi/giava/pyroki/examples")

from webrtc_headset import WebRTCHeadset
from transform_utils import pose2mat
import pyroki_snippets as pks

import logging

logging.basicConfig(
    filename="teleop_timing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def now():
    return time.monotonic()

dataset_finalized = False
camera_shutdown = False

# CONSTANTS 

TIMING_LOG = {
    "loop": [],
    "key": [],
    "headset": [],
    "ik_solve": [],
    "ik_section": [],
    "cmd": [],
    "log": [],
}

URDF_PATH = "/home/devi/giava/right.urdf"
RIGHT_EE_LINK = "rightgripper_base"
DATASET_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot"
TIMING_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/timing_logs"

CAMERA_SERIALS = {
    "right_wrist": "230322270105",
    "top_scene": "230322270396",
}

TASKS = {
    1: "screwdriver_insertion",
    2: "block_square",
}

RIGHT_ARM_NAMES = [
    "rightwaist",
    "rightshoulder",
    "rightelbow",
    "rightforearm_roll",
    "rightwrist_angle",
    "rightwrist_rotate",
]

latest_frames = {name: None for name in CAMERA_SERIALS}
latest_frame_timestamps = {name: None for name in CAMERA_SERIALS}
frame_lock = threading.Lock()
camera_ring_buffers = {name: deque(maxlen=600) for name in CAMERA_SERIALS}

RIGHT_START_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
RIGHT_REST_Q = np.array([0.0, -1.6, 1.5, 0.0, 0.7, 0.0], dtype=float)

from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

GRIPPER_CURRENT_LIMIT = 200

# FUNCTIONS

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

latest_key = None

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

@dataclass
class CameraSample:
    host_ts: float
    frame: np.ndarray
    frame_counter: Optional[int] = None
    hw_ts_ms: Optional[float] = None


@dataclass
class RobotSample:
    obs_ts: float
    joint_positions: np.ndarray            # shape (6,)
    gripper_position: np.ndarray           # shape (1,)
    commanded_joint_positions: np.ndarray  # shape (6,)
    commanded_gripper: np.ndarray          # shape (1,)
    ee_pose: np.ndarray                    # shape (7,) -> [x, y, z, qw, qx, qy, qz]

@dataclass
class EpisodeBuffer:
    task_name: str
    episode_idx: int
    robot_samples: List[RobotSample] = field(default_factory=list)
    camera_samples: Dict[str, List[CameraSample]] = field(default_factory=dict)
    last_camera_frame_counters: Dict[str, Optional[int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.camera_samples:
            self.camera_samples = {name: [] for name in CAMERA_SERIALS}
        if not self.last_camera_frame_counters:
            self.last_camera_frame_counters = {name: None for name in CAMERA_SERIALS}

@dataclass
class TeleopConfig:
    control_dt: float = 1.0 / 30.0
    position_scale: float = 1.35
    alpha: float = 0.3 # previously at 0.22
    arm_cmd_dt: float = 1.0 / 30.0 # previously at 0.016
    moving_time: float = 0.14
    accel_time: float = 0.04
    max_ee_step: float = 0.02 # previously at 0.015
    max_joint_step: np.ndarray = None
    full_joint_velocity_limits_value: float = 2.3
    pos_weight: float = 40.0
    ori_weight: float = 0.25
    dq_weight: float = 0.18
    R_remap_right: np.ndarray = None
    joint_reached_tol: float = 0.03
    ee_reached_tol: float = 0.01
    cmd_timeout: float = 0.25

    def __post_init__(self):
        if self.max_joint_step is None:
            self.max_joint_step = np.array(
                [0.05, 0.05, 0.06, 0.08, 0.08, 0.10],
                dtype=float
            )
        if self.R_remap_right is None:
            self.R_remap_right = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)

@dataclass
class TeleopSessionState:
    active: bool = False
    right_start_controller_pos: np.ndarray = None
    right_start_controller_rot: np.ndarray = None
    right_start_robot_pos: np.ndarray = None
    right_start_robot_rot: np.ndarray = None
    right_filtered_target_pos: np.ndarray = None

@dataclass
class RobotCommandState:
    last_arm_cmd_time: float = 0.0
    last_right_cmd: np.ndarray = None

@dataclass
class CommandKinematicsState:
    q_cmd: np.ndarray = None
    T_right_cmd: np.ndarray = None

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

def update_gripper(
    bot,
    trigger_pressed,
    close_position=-1.5,
    open_position=0.0,
):
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


def camera_worker(name, pipeline):
    frame_count = 0
    stored_count = 0
    last_report = now()

    while not camera_shutdown:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame = digital_zoom(frame, zoom=1.8)

            host_ts = now()
            hw_ts_ms = None
            frame_counter = None

            try:
                hw_ts_ms = float(color_frame.get_timestamp())
            except Exception:
                pass

            try:
                frame_counter = int(color_frame.get_frame_number())
            except Exception:
                pass

            sample = CameraSample(
                host_ts=host_ts,
                frame=frame.copy(),
                frame_counter=frame_counter,
                hw_ts_ms=hw_ts_ms,
            )

            with frame_lock:
                latest_frames[name] = frame.copy()
                latest_frame_timestamps[name] = host_ts
                camera_ring_buffers[name].append(sample)

            frame_count += 1
            stored_count += 1

            if frame_count % 100 == 0:
                logging.info(
                    f"[CAM {name}] buffered={len(camera_ring_buffers[name])}, "
                    f"host_ts={host_ts:.6f}, hw_ts_ms={hw_ts_ms}"
                )

            if stored_count >= 120:
                elapsed = now() - last_report
                rate = stored_count / max(elapsed, 1e-9)
                # print(f"[CAM {name}] stored={stored_count} rate={rate:.1f} Hz buffer={len(camera_ring_buffers[name])}")
                last_report = now()
                stored_count = 0

        except RuntimeError as e:
            if camera_shutdown:
                break
            print(f"{name} camera error: {e}")
            continue

def reset_timing_log():
    for k in TIMING_LOG:
        TIMING_LOG[k].clear()

def get_closest_camera_sample(samples, target_ts, max_dt=0.05):
    if not samples:
        return None
    best = min(samples, key=lambda s: abs(s.host_ts - target_ts))
    if abs(best.host_ts - target_ts) > max_dt:
        return None
    return best

def build_lerobot_frames_from_episode(ep: EpisodeBuffer, max_cam_dt=0.10):
    aligned_frames = []

    for rs in ep.robot_samples:
        cam_matches = {}
        valid = True

        for cam_name, samples in ep.camera_samples.items():
            cam_s = get_closest_camera_sample(samples, rs.obs_ts, max_dt=max_cam_dt)
            if cam_s is None:
                valid = False
                break
            cam_matches[cam_name] = cam_s

        if not valid:
            continue

        obs_state = np.concatenate(
            [rs.joint_positions.astype(np.float32), rs.gripper_position.astype(np.float32)],
            axis=0,
        )
        action = np.concatenate(
            [rs.commanded_joint_positions.astype(np.float32), rs.commanded_gripper.astype(np.float32)],
            axis=0,
        )

        frame = {
            # "task": ep.task_name,
            "observation.state": torch.from_numpy(obs_state),
            "action": torch.from_numpy(action),
            "observation.ee_pose": torch.from_numpy(rs.ee_pose.astype(np.float32)),
            "observation.images.right_wrist": torch.from_numpy(cam_matches["right_wrist"].frame.copy()),
            "observation.images.top_scene": torch.from_numpy(cam_matches["top_scene"].frame.copy()),
            "observation.timestamps.robot": torch.tensor([rs.obs_ts], dtype=torch.float32),
            "observation.timestamps.right_wrist": torch.tensor([cam_matches["right_wrist"].host_ts], dtype=torch.float32),
            "observation.timestamps.top_scene": torch.tensor([cam_matches["top_scene"].host_ts], dtype=torch.float32),
        }
        aligned_frames.append(frame)

    for cam_name, samples in ep.camera_samples.items():
        print(cam_name, "num_samples=", len(samples))

    return aligned_frames

def save_timing_log(task_name, episode_idx):
    if not TIMING_LOG["loop"]:
        print("No timing data to save.")
        return

    out_dir = os.path.join(TIMING_ROOT, task_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timing_episode_{episode_idx:04d}.npz")

    np.savez(
        out_path,
        loop=np.array(TIMING_LOG["loop"], dtype=float),
        key=np.array(TIMING_LOG["key"], dtype=float),
        headset=np.array(TIMING_LOG["headset"], dtype=float),
        ik_solve=np.array(TIMING_LOG["ik_solve"], dtype=float),
        ik_section=np.array(TIMING_LOG["ik_section"], dtype=float),
        cmd=np.array(TIMING_LOG["cmd"], dtype=float),
        log=np.array(TIMING_LOG["log"], dtype=float),
    )
    print(f"Saved timing log to {out_path}")

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
    for name, serial in CAMERA_SERIALS.items():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        threading.Thread(target=camera_worker, args=(name, pipeline), daemon=True).start()
        pipelines[name] = pipeline
        print(f"Started: {name}")
    return pipelines

def start_episode_buffer(task_name, episode_idx):
    return EpisodeBuffer(task_name=task_name, episode_idx=episode_idx)

def finalize_episode_to_dataset(dataset, episode_buffer, task_name, episode_idx, max_cam_dt=0.05):
    aligned_frames = build_lerobot_frames_from_episode(episode_buffer, max_cam_dt=max_cam_dt)

    print(
        f"Episode {episode_idx}: "
        f"{len(episode_buffer.robot_samples)} robot samples, "
        f"{len(aligned_frames)} aligned frames"
    )

    if len(aligned_frames) == 0:
        print(f"No aligned frames found for episode {episode_idx}.")
        return False

    for frame in aligned_frames:
        dataset.add_frame(frame, episode_buffer.task_name)

    dataset.save_episode()
    save_timing_log(task_name, episode_idx)
    reset_timing_log()
    print(f"Saved episode {episode_idx} with {len(aligned_frames)} aligned frames.")
    return True

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
    episode_buffer,
    episode_has_frames,
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
):
    global dataset_finalized
    global camera_shutdown
    # 1. Handle active episode
    if collecting_episode and episode_buffer is not None:
        if episode_has_frames:
            save = ask_yes_no(
                "\nAn episode is currently being recorded. Save it before quitting?",
                default=True,
            )
            if save:
                ok = finalize_episode_to_dataset(
                    dataset=dataset,
                    episode_buffer=episode_buffer,
                    task_name=task_name,
                    episode_idx=episode_idx,
                    max_cam_dt=0.10,
                )
                if ok:
                    print("\nEPISODE SAVED ON QUIT")
                    episode_idx += 1
                    dataset_finalized = False
            else:
                print("\nDiscarding current in-memory episode on quit.")
        else:
            print("\nNo frames in current episode. Discarding.")

        collecting_episode = False
        episode_buffer = None

    # 2. Move robot to rest pose
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

    # 3. Consolidate dataset if needed
    if not dataset_finalized:
        try:
            dataset.consolidate()
            dataset_finalized = True
            print("\nDATASET CONSOLIDATED ON EXIT")
        except Exception as e:
            print(f"Consolidate failed on exit: {e}")

    # 4. Stop cameras
    camera_shutdown = True
    for pipeline in pipelines.values():
        try:
            pipeline.stop()
        except Exception as e:
            print(f"Error stopping camera pipeline: {e}")

    print("\nShutdown complete.")
    return episode_idx

def main():
    episode_buffer = None
    collecting_episode = False
    global latest_key, dataset_finalized
    episode_has_frames = False
    cfg = TeleopConfig()

    # 1. Init ROS
    rospy.init_node("bimanual_vr_teleop")

    # 2. Print tasks and get selection BEFORE starting headset / cameras / keyboard
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
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # camera pipelines
    pipelines = setup_cameras()

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
                    "rightwaist",
                    "rightshoulder",
                    "rightelbow",
                    "rightforearm_roll",
                    "rightwrist_angle",
                    "rightwrist_rotate",
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
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    while not rospy.is_shutdown():
        loop_start = now()

        t_key_start = now()

        key = latest_key
        latest_key = None

        if key == "s":
            try:
                if not collecting_episode or episode_buffer is None:
                    print("\nNo active episode to save.")
                elif not episode_has_frames:
                    print("\nNo frames recorded for this episode.")
                else:
                    ok = finalize_episode_to_dataset(
                        dataset=dataset,
                        episode_buffer=episode_buffer,
                        task_name=task_name,
                        episode_idx=episode_idx,
                        max_cam_dt=0.10,
                    )
                    if ok:
                        print("\nEPISODE SAVED")
                        episode_idx += 1
                        dataset_finalized = False

                    collecting_episode = False
                    episode_buffer = None
                    episode_has_frames = False
            except Exception as e:
                print(f"\nEpisode save failed: {e}")


        if key == "i":
            stop_teleop_session(teleop_state)

            if collecting_episode:
                print("\nDiscarding current in-memory episode before resetting to start pose.")
                collecting_episode = False
                episode_buffer = None
                episode_has_frames = False
                reset_timing_log()

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
            t_key_elapsed = now() - t_key_start
            print(f"[TIMING] key section: {t_key_elapsed*1000:.2f} ms")
            continue

        if key == "d":
            if collecting_episode:
                collecting_episode = False
                episode_buffer = None
                episode_has_frames = False
                reset_timing_log()
                print("\nEPISODE DISCARDED")

        if key == "q":
            print("\nExiting...")

            episode_idx = safe_shutdown(
                right_bot=right_bot,
                pipelines=pipelines,
                dataset=dataset,
                collecting_episode=collecting_episode,
                episode_buffer=episode_buffer,
                episode_has_frames=episode_has_frames,
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
            )

            break


        if key == "r" and not collecting_episode:
            t_rec_start = now()
            print("\nCOLLECTING NEW EPISODE")

            for cam_name in CAMERA_SERIALS:
                camera_ring_buffers[cam_name].clear()

            episode_buffer = start_episode_buffer(task_name, episode_idx)
            collecting_episode = True
            episode_has_frames = False
            reset_timing_log()

            print(f"\nCOLLECTING: episode_{episode_idx:04d}")

            t_rec_elapsed = now() - t_rec_start
            logging.info(f"[TIMING] start episode section: {t_rec_elapsed*1000:.2f} ms")

        t_key_elapsed = now() - t_key_start

        t_headset_start = now()

        headset_data = headset.receive_data()
        if headset_data is None:
            time.sleep(0.01)
            t_headset_elapsed = now() - t_headset_start
            t_loop_elapsed = now() - loop_start
            logging.info(f"[TIMING] headset none, headset={t_headset_elapsed*1000:.2f} ms, loop={t_loop_elapsed*1000:.2f} ms")
            continue

        right_controller = pose2mat(headset_data.r_pos, headset_data.r_quat)
        button_pressed = headset_data.r_button_one
        right_trigger = headset_data.r_index_trigger

        t_headset_elapsed = now() - t_headset_start

        t_ik_start = now()

        if button_pressed and not teleop_state.active:
            start_teleop_session(teleop_state, right_controller, cmd_kin.T_right_cmd)
            print("\nTeleop ENABLED")
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")

        right_gripper_action = update_gripper(right_bot, right_trigger > 0)

        if teleop_state.active:
            right_target_pos, right_target_wxyz = compute_right_target(
                cfg,
                teleop_state,
                right_controller,
                cmd_kin.T_right_cmd,
            )

            t_solve_start = now()
            if (t_solve_start - cmd_state.last_arm_cmd_time) >= cfg.arm_cmd_dt:
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
                t_solve_elapsed = now() - t_solve_start

                if q_new is not None:
                    right_target_q = np.asarray(q_new[right_arm_indices], dtype=float)
                    right_prev_cmd = (
                        cmd_state.last_right_cmd
                        if cmd_state.last_right_cmd is not None
                        else cmd_kin.q_cmd[right_arm_indices]
                    )

                    t_cmd_start = now()
                    right_q_cmd = clamp_joint_step(right_prev_cmd, right_target_q, cfg.max_joint_step)

                    right_bot.arm.set_joint_positions(
                        right_q_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )
                    t_cmd_elapsed = now() - t_cmd_start

                    cmd_kin.q_cmd[right_arm_indices] = right_q_cmd

                    fk_cmd = robot.forward_kinematics(cmd_kin.q_cmd)
                    cmd_kin.T_right_cmd = jaxlie.SE3(fk_cmd[right_ee_index]).as_matrix()

                    cmd_state.last_right_cmd = right_q_cmd.copy()
                    cmd_state.last_arm_cmd_time = now()

                    TIMING_LOG["ik_solve"].append(t_solve_elapsed)
                    TIMING_LOG["cmd"].append(t_cmd_elapsed)

                    logging.info(
                        f"[TIMING] ik={t_solve_elapsed*1000:.2f} ms, cmd={t_cmd_elapsed*1000:.2f} ms"
                    )

        t_ik_elapsed = now() - t_ik_start

        t_log_start = now()

        right_joint_state_msg = right_bot.dxl.joint_states
        right_joint_positions = np.array(right_joint_state_msg.position[:6], dtype=np.float32)
        right_gripper_state = np.array([right_joint_state_msg.position[6]], dtype=np.float32)

        if collecting_episode and episode_buffer is not None:
            obs_ts = now()

            commanded_joint_positions = (
                right_joint_positions.copy()
                if cmd_state.last_right_cmd is None
                else np.asarray(cmd_state.last_right_cmd, dtype=np.float32).copy()
            )
            commanded_gripper = np.array([right_gripper_action], dtype=np.float32)

            q_measured = np.zeros(robot.joints.num_actuated_joints, dtype=float)
            q_measured[right_arm_indices] = right_joint_positions.astype(float)
            _, T_right_measured = refresh_fk(robot, q_measured, right_ee_index)
            ee_pose = matrix_to_pose7(T_right_measured)

            episode_buffer.robot_samples.append(
                RobotSample(
                    obs_ts=obs_ts,
                    joint_positions=right_joint_positions.copy(),
                    gripper_position=right_gripper_state.copy(),
                    commanded_joint_positions=commanded_joint_positions.copy(),
                    commanded_gripper=commanded_gripper.copy(),
                    ee_pose=ee_pose.copy(),
                )
            )

            with frame_lock:
                for cam_name in CAMERA_SERIALS:
                    if len(camera_ring_buffers[cam_name]) > 0:
                        newest = camera_ring_buffers[cam_name][-1]
                        last_ctr = episode_buffer.last_camera_frame_counters[cam_name]

                        if newest.frame_counter != last_ctr:
                            episode_buffer.camera_samples[cam_name].append(
                                CameraSample(
                                    host_ts=newest.host_ts,
                                    frame=newest.frame.copy(),
                                    frame_counter=newest.frame_counter,
                                    hw_ts_ms=newest.hw_ts_ms,
                                )
                            )
                            episode_buffer.last_camera_frame_counters[cam_name] = newest.frame_counter

            episode_has_frames = True

        t_log_elapsed = now() - t_log_start

        loop_elapsed = now() - loop_start
        sleep_time = max(0, cfg.control_dt - loop_elapsed)

        TIMING_LOG["loop"].append(loop_elapsed)
        TIMING_LOG["key"].append(t_key_elapsed)
        TIMING_LOG["headset"].append(t_headset_elapsed)
        TIMING_LOG["ik_section"].append(t_ik_elapsed)
        TIMING_LOG["log"].append(t_log_elapsed)

        if loop_elapsed > cfg.control_dt:
            logging.info(
                f"[TIMING] overrun: loop={loop_elapsed*1000:.2f} ms, "
                f"key={t_key_elapsed*1000:.2f} ms, "
                f"headset={t_headset_elapsed*1000:.2f} ms, "
                f"ik_section={t_ik_elapsed*1000:.2f} ms, "
                f"log={t_log_elapsed*1000:.2f} ms"
            )

        time.sleep(sleep_time)

if __name__ == "__main__":
    main()