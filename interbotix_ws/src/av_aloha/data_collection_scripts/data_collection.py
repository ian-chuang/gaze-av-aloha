import os
import sys
import time
import threading
from dataclasses import dataclass
from queue import Full, Empty, Queue

import cv2
import h5py
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

def now():
    # monotonic is ideal for elapsed timings
    return time.monotonic()

TIMING_LOG = {
    "loop": [],
    "key": [],
    "headset": [],
    "ik_solve": [],
    "ik_section": [],
    "cmd": [],
    "log": [],
}

URDF_PATH = "/home/devi/giava/giava.urdf"
LEFT_EE_LINK = "leftgripper_base"
RIGHT_EE_LINK = "rightgripper_base"
DATASET_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset"

CAMERA_SERIALS = {
    "left_wrist": "230322272239",
    "right_wrist": "230322270105",
    "top_scene": "230322270396",
}

TASKS = {
    1: "pick_up_cup",
    2: "stack_blocks",
    3: "open_drawer",
}

LEFT_ARM_NAMES = [
    "leftwaist",
    "leftshoulder",
    "leftelbow",
    "leftforearm_roll",
    "leftwrist_angle",
    "leftwrist_rotate",
]

RIGHT_ARM_NAMES = [
    "rightwaist",
    "rightshoulder",
    "rightelbow",
    "rightforearm_roll",
    "rightwrist_angle",
    "rightwrist_rotate",
]

MIDDLE_ARM_NAMES = [
    "middlebase_link",
    "middleshoulder_link",
    "middleupper_arm_link",
    "middleupper_forearm_link",
    "middlelower_forearm_link",
    "middlewrist_link",
    "middlepan_link",
]

LEFT_RESET_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
RIGHT_RESET_Q = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
MIDDLE_INIT_Q = np.array([0.05, -1.5, 0.05, -0.08, 2.0, 1.4, 0.0], dtype=float)
       # 12.5 Hz solve/publish loop

# POS_WEIGHT = 40.0
# ORI_WEIGHT = 0.5           # or 0.0 if you can tolerate fixed wrist orientation
# DQ_WEIGHT = 0.15

# MAX_EE_STEP = 0.012        # meters per IK solve
# MAX_JOINT_STEP = np.array([0.03, 0.03, 0.04, 0.06, 0.06, 0.08])

# ARM_MOVING_TIME = 0.18
# ARM_ACCEL_TIME = 0.06
# ALPHA = 0.3
# POSITION_SCALE = 2.0

latest_key = None
recording = False
frame_queues = {name: Queue(maxsize=30) for name in CAMERA_SERIALS}
writer_threads = {}

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

@dataclass
class TeleopConfig:
    control_dt: float = 0.05
    position_scale: float = 2.0
    alpha: float = 0.3
    arm_cmd_dt: float = 0.10
    moving_time: float = 0.20
    accel_time: float = 0.08
    max_ee_step: float = 0.010
    max_joint_step: np.ndarray = None
    full_joint_velocity_limits_value: float = 3.0
    pos_weight: float = 40.0
    ori_weight: float = 0.5
    dq_weight: float = 0.15
    R_remap_left: np.ndarray = None
    R_remap_right: np.ndarray = None

    def __post_init__(self):
        if self.max_joint_step is None:
            self.max_joint_step = np.array([0.03, 0.03, 0.04, 0.06, 0.06, 0.08], dtype=float)
        if self.R_remap_left is None:
            self.R_remap_left = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        if self.R_remap_right is None:
            self.R_remap_right = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


@dataclass
class TeleopSessionState:
    active: bool = False
    left_start_controller_pos: np.ndarray = None
    right_start_controller_pos: np.ndarray = None
    left_start_controller_rot: np.ndarray = None
    right_start_controller_rot: np.ndarray = None
    left_start_robot_pos: np.ndarray = None
    right_start_robot_pos: np.ndarray = None
    left_start_robot_rot: np.ndarray = None
    right_start_robot_rot: np.ndarray = None
    left_filtered_target_pos: np.ndarray = None
    right_filtered_target_pos: np.ndarray = None


@dataclass
class RobotCommandState:
    last_arm_cmd_time: float = 0.0
    last_left_cmd: np.ndarray = None
    last_right_cmd: np.ndarray = None

@dataclass
class CommandKinematicsState:
    q_cmd: np.ndarray = None
    T_left_cmd: np.ndarray = None
    T_right_cmd: np.ndarray = None

def publish_grippers(left_bot, right_bot, left_trigger, right_trigger):
    left_cmd = JointSingleCommand(name="gripper")
    right_cmd = JointSingleCommand(name="gripper")
    left_cmd.cmd = -1.7 if left_trigger > 0 else 0.1
    right_cmd.cmd = -1.7 if right_trigger > 0 else 0.1
    left_bot.gripper.core.pub_single.publish(left_cmd)
    right_bot.gripper.core.pub_single.publish(right_cmd)
    return left_cmd.cmd, right_cmd.cmd


def start_teleop_session(state, left_controller, right_controller, T_left, T_right):
    state.active = True
    state.left_start_controller_pos = left_controller[:3, 3].copy()
    state.right_start_controller_pos = right_controller[:3, 3].copy()
    state.left_start_controller_rot = left_controller[:3, :3].copy()
    state.right_start_controller_rot = right_controller[:3, :3].copy()
    state.left_start_robot_pos = T_left[:3, 3].copy()
    state.right_start_robot_pos = T_right[:3, 3].copy()
    state.left_start_robot_rot = T_left[:3, :3].copy()
    state.right_start_robot_rot = T_right[:3, :3].copy()
    state.left_filtered_target_pos = T_left[:3, 3].copy()
    state.right_filtered_target_pos = T_right[:3, 3].copy()


def stop_teleop_session(state):
    state.active = False


def compute_bimanual_targets(cfg, state, left_controller, right_controller, T_left, T_right):
    left_delta_ctrl = left_controller[:3, 3] - state.left_start_controller_pos
    right_delta_ctrl = right_controller[:3, 3] - state.right_start_controller_pos

    left_delta_robot = cfg.R_remap_left @ left_delta_ctrl
    right_delta_robot = cfg.R_remap_right @ right_delta_ctrl

    left_raw = state.left_start_robot_pos + cfg.position_scale * left_delta_robot
    right_raw = state.right_start_robot_pos + cfg.position_scale * right_delta_robot

    state.left_filtered_target_pos = cfg.alpha * left_raw + (1.0 - cfg.alpha) * state.left_filtered_target_pos
    state.right_filtered_target_pos = cfg.alpha * right_raw + (1.0 - cfg.alpha) * state.right_filtered_target_pos

    left_target_pos = clamp_cartesian_step(state.left_filtered_target_pos, T_left[:3, 3].copy(), cfg.max_ee_step)
    right_target_pos = clamp_cartesian_step(state.right_filtered_target_pos, T_right[:3, 3].copy(), cfg.max_ee_step)

    left_delta_rot = left_controller[:3, :3] @ state.left_start_controller_rot.T
    right_delta_rot = right_controller[:3, :3] @ state.right_start_controller_rot.T

    left_target_rot = left_delta_rot @ state.left_start_robot_rot
    right_target_rot = right_delta_rot @ state.right_start_robot_rot

    left_target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(left_target_rot).as_quat())
    right_target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(right_target_rot).as_quat())

    return left_target_pos, right_target_pos, left_target_wxyz, right_target_wxyz


# def maybe_send_arm_commands(cfg, cmd_state, now, q_new, q, left_arm_indices, right_arm_indices, left_bot, right_bot):
#     if q_new is None:
#         return False, "ik_failed"
#     if (now - cmd_state.last_arm_cmd_time) < cfg.arm_cmd_dt:
#         return False, "rate_limited"

#     left_measured = np.array(left_bot.dxl.joint_states.position[:6], dtype=float)
#     right_measured = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)

#     left_target_q = np.array(q_new[left_arm_indices], dtype=float)
#     right_target_q = np.array(q_new[right_arm_indices], dtype=float)

#     left_safe_q = clamp_joint_step(left_measured, left_target_q, cfg.max_joint_step)
#     right_safe_q = clamp_joint_step(right_measured, right_target_q, cfg.max_joint_step)

#     left_bot.arm.set_joint_positions(left_safe_q.tolist(), moving_time=cfg.moving_time, accel_time=cfg.accel_time, blocking=False)
#     right_bot.arm.set_joint_positions(right_safe_q.tolist(), moving_time=cfg.moving_time, accel_time=cfg.accel_time, blocking=False)

#     q[left_arm_indices] = left_safe_q
#     q[right_arm_indices] = right_safe_q
#     cmd_state.last_left_cmd = left_safe_q.copy()
#     cmd_state.last_right_cmd = right_safe_q.copy()
#     cmd_state.last_arm_cmd_time = now
#     return True, "sent"


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


# def move_to_reset_pose(left_bot, right_bot):
#     print("\nMoving LEFT arm safely...")
#     safe_move_arm_joints(left_bot, LEFT_RESET_Q, total_time=4.0, step_time=0.3)
#     print("\nMoving RIGHT arm safely...")
#     safe_move_arm_joints(right_bot, RIGHT_RESET_Q, total_time=4.0, step_time=0.3)
    
def move_to_reset_pose(
    left_bot,
    right_bot,
    q,
    cmd_kin,
    cmd_state,
    left_arm_indices,
    right_arm_indices,
    middle_arm_indices,
    robot,
    left_ee_index,
    right_ee_index,
    cfg,
    full_joint_velocity_limits,
):
    print("\nMoving LEFT arm safely...")
    safe_move_arm_joints(left_bot, LEFT_RESET_Q, total_time=4.0, step_time=0.3)
    print("\nMoving RIGHT arm safely...")
    safe_move_arm_joints(right_bot, RIGHT_RESET_Q, total_time=4.0, step_time=0.3)

    # Update global commanded state
    q[left_arm_indices] = LEFT_RESET_Q.copy()
    q[right_arm_indices] = RIGHT_RESET_Q.copy()
    q[middle_arm_indices] = MIDDLE_INIT_Q.copy()

    _, T_left, T_right = refresh_fk(robot, q, left_ee_index, right_ee_index)

    cmd_kin.q_cmd = q.copy()
    cmd_kin.T_left_cmd = T_left.copy()
    cmd_kin.T_right_cmd = T_right.copy()
    cmd_state.last_left_cmd = LEFT_RESET_Q.copy()
    cmd_state.last_right_cmd = RIGHT_RESET_Q.copy()

    cmd_state.last_arm_cmd_time = 0.0  # reset command timer so next teleop is allowed immediately

    # _ = pks.solve_trajectories_ik(
    #     robot=robot,
    #     target_link_names=[LEFT_EE_LINK, RIGHT_EE_LINK],
    #     target_wxyzs=[
    #         quat_xyzw_to_wxyz(R.from_matrix(T_left[:3, :3]).as_quat()),
    #         quat_xyzw_to_wxyz(R.from_matrix(T_right[:3, :3]).as_quat()),
    #     ],
    #     target_positions=[T_left[:3, 3], T_right[:3, 3]],
    #     prev_q=cmd_kin.q_cmd,
    #     dt=cfg.arm_cmd_dt,
    #     joint_velocity_limits=full_joint_velocity_limits,
    #     pos_weight=cfg.pos_weight,
    #     ori_weight=cfg.ori_weight,
    #     dq_weight=cfg.dq_weight,
    # )

def append_episode_step(episode_data, left_joint_positions, right_joint_positions, left_gripper_state, right_gripper_state,
                        cmd_state, left_gripper_action, right_gripper_action):
    if cmd_state.last_left_cmd is None or cmd_state.last_right_cmd is None:
        left_action_arm = left_joint_positions.copy()
        right_action_arm = right_joint_positions.copy()
    else:
        left_action_arm = cmd_state.last_left_cmd.copy()
        right_action_arm = cmd_state.last_right_cmd.copy()

    episode_data.append({
        "timestamp": time.monotonic(),
        "left_q": np.concatenate([left_joint_positions, left_gripper_state]),
        "right_q": np.concatenate([right_joint_positions, right_gripper_state]),
        "left_action": np.concatenate([left_action_arm, [left_gripper_action]]),
        "right_action": np.concatenate([right_action_arm, [right_gripper_action]]),
    })


def keyboard_listener():
    global latest_key
    while True:
        latest_key = input().strip()


def camera_worker(name, pipeline):
    global recording, frame_queues
    frame_count = 0
    while True:
        try:
            t_wait_start = now()
            frames = pipeline.wait_for_frames()
            t_wait_elapsed = now() - t_wait_start

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            t_convert_start = now()
            frame = np.asanyarray(color_frame.get_data())
            t_convert_elapsed = now() - t_convert_start

            timestamp = now()
            if recording:
                try:
                    frame_queues[name].put_nowait((timestamp, frame.copy()))
                    frame_count += 1
                    if frame_count % 50 == 0:
                        logging.info(
                            f"[CAM {name}] wait={t_wait_elapsed*1000:.2f} ms, "
                            f"convert={t_convert_elapsed*1000:.2f} ms"
                        )
                except Full:
                    # optional: log drops
                    print(f"[CAM {name}] frame queue full, dropping frame")
                    pass
        except RuntimeError as e:
            print(f"{name} camera error: {e}")
            continue


def video_writer_worker(name, output_path):
    global recording, frame_queues
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
    timestamps = []
    frame_count = 0

    while True:
        try:
            t_get_start = now()
            item = frame_queues[name].get(timeout=1.0)
            t_get_elapsed = now() - t_get_start
        except Empty:
            if not recording:
                break
            continue

        if item is None:
            break

        timestamp, frame = item

        t_write_start = now()
        writer.write(frame)
        t_write_elapsed = now() - t_write_start

        timestamps.append(timestamp)
        frame_count += 1

        if frame_count % 50 == 0:
            logging.info(
                f"[WRITER {name}] get={t_get_elapsed*1000:.2f} ms, "
                f"write={t_write_elapsed*1000:.2f} ms"
            )

    writer.release()
    np.save(output_path.replace(".mp4", "_timestamps.npy"), np.array(timestamps))
    print(f"Saved timestamps for {name}")


def start_episode():
    return []


def save_episode(episode_data, task_name, episode_idx, control_dt):
    dataset_dir = os.path.join(DATASET_ROOT, task_name)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx:04d}.hdf5")

    max_timesteps = len(episode_data)
    if max_timesteps == 0:
        print("Empty episode")
        return

    with h5py.File(dataset_path, "w") as root:
        root.attrs["fps"] = int(1 / control_dt)
        root.attrs["control_dt"] = control_dt

        obs = root.create_group("observations")
        first_step = episode_data[0]

        qpos_dim = len(np.concatenate([first_step["left_q"], first_step["right_q"]]))
        action_dim = len(np.concatenate([first_step["left_action"], first_step["right_action"]]))

        qpos = obs.create_dataset("qpos", (max_timesteps, qpos_dim), dtype=np.float32)
        action = root.create_dataset("action", (max_timesteps, action_dim), dtype=np.float32)
        timestamps = root.create_dataset("timestamp", (max_timesteps,), dtype=np.float64)

        for idx, step in enumerate(episode_data):
            qpos[idx] = np.concatenate([step["left_q"], step["right_q"]])
            action[idx] = np.concatenate([step["left_action"], step["right_action"]])
            timestamps[idx] = step["timestamp"]

    print(f"\nSaved: {dataset_path}")

def save_timing_log(task_name, episode_idx):
    if not TIMING_LOG["loop"]:
        print("No timing data to save.")
        return

    out_dir = os.path.join(DATASET_ROOT, task_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timing_episode_{episode_idx:04d}.npz")

    # convert lists to numpy arrays
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

    left_arm_indices = [robot.joints.actuated_names.index(name) for name in LEFT_ARM_NAMES]
    right_arm_indices = [robot.joints.actuated_names.index(name) for name in RIGHT_ARM_NAMES]
    middle_arm_indices = [robot.joints.actuated_names.index(name) for name in MIDDLE_ARM_NAMES]

    left_ee_index = robot.links.names.index(LEFT_EE_LINK)
    right_ee_index = robot.links.names.index(RIGHT_EE_LINK)

    return robot, left_arm_indices, right_arm_indices, middle_arm_indices, left_ee_index, right_ee_index


def create_bots(cfg):
    left_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=cfg.moving_time,
        accel_time=cfg.accel_time,
        init_node=False,
    )
    left_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    left_bot.dxl.robot_torque_enable("single", "gripper", True)

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=cfg.moving_time,
        accel_time=cfg.accel_time,
        init_node=False,
    )
    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    middle_bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="puppet_middle",
        moving_time=2.0,
        accel_time=0.5,
        init_node=False,
    )
    middle_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    middle_bot.dxl.robot_torque_enable("group", "arm", True)

    return left_bot, right_bot, middle_bot

# def solve_single_arm_ik(
#     robot,
#     target_link_name,
#     target_position,
#     target_wxyz,
#     prev_q,
#     dt,
#     joint_velocity_limits,
#     pos_weight=40.0,
#     ori_weight=0.5,
#     dq_weight=0.15,
# ):
#     return pks.solve_trajectories_ik(
#         robot=robot,
#         target_link_names=[target_link_name],
#         target_positions=[target_position],
#         target_wxyzs=[target_wxyz],
#         prev_q=prev_q,
#         dt=dt,
#         joint_velocity_limits=joint_velocity_limits,
#         pos_weight=pos_weight,
#         ori_weight=ori_weight,
#         dq_weight=dq_weight,
#     )

def initialize_bots(left_bot, right_bot, middle_bot):
    left_bot.arm.set_joint_positions(LEFT_RESET_Q.tolist(), moving_time=2.0, accel_time=0.5, blocking=True)
    right_bot.arm.set_joint_positions(RIGHT_RESET_Q.tolist(), moving_time=2.0, accel_time=0.5, blocking=True)
    middle_bot.arm.set_joint_positions(MIDDLE_INIT_Q[:6].tolist(), moving_time=2.0, accel_time=0.5, blocking=True)
    time.sleep(2)
    rospy.sleep(2.0)
    rospy.sleep(1.0)


def setup_cameras():
    pipelines = {}
    for name, serial in CAMERA_SERIALS.items():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipeline.start(config)
        threading.Thread(target=camera_worker, args=(name, pipeline), daemon=True).start()
        pipelines[name] = pipeline
        print(f"Started: {name}")
    return pipelines


def stop_recording_and_flush():
    global recording, frame_queues, writer_threads
    recording = False
    for q in frame_queues.values():
        try:
            q.put_nowait(None)
        except Full:
            pass
    for t in writer_threads.values():
        t.join()
    print("Video writers stopped.")


def refresh_fk(robot, q, left_ee_index, right_ee_index):
    fk = robot.forward_kinematics(q)
    T_left = jaxlie.SE3(fk[left_ee_index]).as_matrix()
    T_right = jaxlie.SE3(fk[right_ee_index]).as_matrix()
    return fk, T_left, T_right


def main():
    global latest_key, recording, frame_queues, writer_threads

    cfg = TeleopConfig()

    rospy.init_node("bimanual_vr_teleop")
    threading.Thread(target=keyboard_listener, daemon=True).start()

    headset = WebRTCHeadset()
    headset.run_in_thread()

    pipelines = setup_cameras()
    robot, left_arm_indices, right_arm_indices, middle_arm_indices, left_ee_index, right_ee_index = build_robot_model()
    left_bot, right_bot, middle_bot = create_bots(cfg)
    initialize_bots(left_bot, right_bot, middle_bot)

    q = np.zeros(robot.joints.num_actuated_joints)
    q[left_arm_indices] = np.array(left_bot.dxl.joint_states.position[:6], dtype=float)
    q[right_arm_indices] = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)
    q[middle_arm_indices] = np.array(MIDDLE_INIT_Q, dtype=float)

    fk, T_left, T_right = refresh_fk(robot, q, left_ee_index, right_ee_index)
    cmd_kin = CommandKinematicsState(q_cmd=q.copy(), T_left_cmd=T_left.copy(), T_right_cmd=T_right.copy())
    full_joint_velocity_limits = np.ones(robot.joints.num_actuated_joints) * cfg.full_joint_velocity_limits_value

    solve_bimanual_ik, warmup_bimanual_ik, _ = pks.make_bimanual_ik_solver(
        robot=robot,
        left_target_link_name=LEFT_EE_LINK,
        right_target_link_name=RIGHT_EE_LINK,
    )

    warmup_bimanual_ik(
        prev_q=cmd_kin.q_cmd.copy(),
        joint_velocity_limits=full_joint_velocity_limits,
        dt=cfg.arm_cmd_dt,
        pos_weight=cfg.pos_weight,
        ori_weight=cfg.ori_weight,
        dq_weight=cfg.dq_weight,
    )

    teleop_state = TeleopSessionState()
    cmd_state = RobotCommandState(
        last_left_cmd=np.array(left_bot.dxl.joint_states.position[:6], dtype=float),
        last_right_cmd=np.array(right_bot.dxl.joint_states.position[:6], dtype=float),
    )

    print("\nREADY")
    print("\nAVAILABLE TASKS:\n")
    for idx, name in TASKS.items():
        print(f"{idx}: {name}")

    task_idx = int(input("\nSelect task number: "))
    task_name = TASKS[task_idx]
    print(f"\nSelected task: {task_name}")

    episode_idx = 0
    episode_data = []
    left_gripper_action = 0.1
    right_gripper_action = 0.1

    while not rospy.is_shutdown():
        loop_start = now()

        # --- SECTION 1: keyboard / episode control ---
        t_key_start = now()

        key = latest_key
        latest_key = None

        if key == "s":
            stop_recording_and_flush()
            save_episode(episode_data, task_name, episode_idx - 1, cfg.control_dt)
            save_timing_log(task_name, episode_idx - 1)
            print("\nEPISODE SAVED")

        if key == "d":
            episode_data = []
            episode_idx -= 1
            recording = False
            print("\nEPISODE NOT SAVED")

        if key == "i":
            stop_teleop_session(teleop_state)
            recording = False
            print("\nTeleop DISABLED. MOVING TO RESET POSE")
            move_to_reset_pose(
                left_bot,
                right_bot,
                q,
                cmd_kin,
                cmd_state,
                left_arm_indices,
                right_arm_indices,
                middle_arm_indices,
                robot,
                left_ee_index,
                right_ee_index,
                cfg,
                full_joint_velocity_limits,
            )
            print("\nRobot reset.")
            t_key_elapsed = now() - t_key_start
            print(f"[TIMING] key section: {t_key_elapsed*1000:.2f} ms")
            continue

        if key == "q":
            print("\nExiting...")
            stop_recording_and_flush()
            for pipeline in pipelines.values():
                pipeline.stop()
            save_timing_log(task_name, episode_idx)
            break

        if key == "r" and not recording:
            t_rec_start = now()
            print("\nRECORDING NEW EPISODE")
            frame_queues = {name: Queue(maxsize=30) for name in CAMERA_SERIALS}
            episode_name = f"episode_{episode_idx:04d}"
            episode_data = start_episode()
            episode_dir = os.path.join(DATASET_ROOT, task_name, episode_name)
            os.makedirs(episode_dir, exist_ok=True)
            writer_threads = {}

            for cam_name in CAMERA_SERIALS.keys():
                video_path = os.path.join(episode_dir, f"{cam_name}.mp4")
                t = threading.Thread(target=video_writer_worker, args=(cam_name, video_path), daemon=True)
                t.start()
                writer_threads[cam_name] = t

            print(f"Started video writers for {episode_name}")
            recording = True
            print(f"\nRECORDING: {episode_name}")
            episode_idx += 1
            t_rec_elapsed = now() - t_rec_start
            logging.info(f"[TIMING] start recording section: {t_rec_elapsed*1000:.2f} ms")

        t_key_elapsed = now() - t_key_start

        # --- SECTION 2: read headset + mapping ---
        t_headset_start = now()

        headset_data = headset.receive_data()
        if headset_data is None:
            time.sleep(0.01)
            # small early exit timing
            t_headset_elapsed = now() - t_headset_start
            t_loop_elapsed = now() - loop_start
            logging.info(f"[TIMING] headset none, headset={t_headset_elapsed*1000:.2f} ms, loop={t_loop_elapsed*1000:.2f} ms")
            continue

        left_controller = pose2mat(headset_data.l_pos, headset_data.l_quat)
        right_controller = pose2mat(headset_data.r_pos, headset_data.r_quat)
        button_pressed = headset_data.r_button_one
        left_trigger = headset_data.l_index_trigger
        right_trigger = headset_data.r_index_trigger

        t_headset_elapsed = now() - t_headset_start

        # --- SECTION 3: teleop session + IK + arm commands ---
        t_ik_start = now()

        if button_pressed and not teleop_state.active:
            start_teleop_session(teleop_state, left_controller, right_controller, cmd_kin.T_left_cmd, cmd_kin.T_right_cmd)
            print("\nTeleop ENABLED")
        elif (not button_pressed) and teleop_state.active:
            stop_teleop_session(teleop_state)
            print("\nTeleop DISABLED")

        # NOTE: you currently define these inside the loop; that’s fine for now, we just time around IK
        left_gripper_action, right_gripper_action = publish_grippers(
            left_bot, right_bot, left_trigger, right_trigger
        )

        if teleop_state.active:
            left_target_pos, right_target_pos, left_target_wxyz, right_target_wxyz = compute_bimanual_targets(
                cfg,
                teleop_state,
                left_controller,
                right_controller,
                cmd_kin.T_left_cmd,
                cmd_kin.T_right_cmd,
            )

            t_solve_start = now()
            if (t_solve_start - cmd_state.last_arm_cmd_time) >= cfg.arm_cmd_dt:
                q_new = solve_bimanual_ik(
                    left_target_position=left_target_pos,
                    right_target_position=right_target_pos,
                    left_target_wxyz=left_target_wxyz,
                    right_target_wxyz=right_target_wxyz,
                    prev_q=cmd_kin.q_cmd,
                    dt=cfg.arm_cmd_dt,
                    joint_velocity_limits=full_joint_velocity_limits,
                    pos_weight=cfg.pos_weight,
                    ori_weight=cfg.ori_weight,
                    dq_weight=cfg.dq_weight,
                    block_until_ready=True,
                )
                t_solve_elapsed = now() - t_solve_start

                if q_new is not None:
                    left_target_q = np.asarray(q_new[left_arm_indices], dtype=float)
                    right_target_q = np.asarray(q_new[right_arm_indices], dtype=float)

                    left_prev_cmd = cmd_state.last_left_cmd if cmd_state.last_left_cmd is not None else cmd_kin.q_cmd[left_arm_indices]
                    right_prev_cmd = cmd_state.last_right_cmd if cmd_state.last_right_cmd is not None else cmd_kin.q_cmd[right_arm_indices]

                    t_cmd_start = now()
                    left_q_cmd = clamp_joint_step(left_prev_cmd, left_target_q, cfg.max_joint_step)
                    right_q_cmd = clamp_joint_step(right_prev_cmd, right_target_q, cfg.max_joint_step)

                    left_bot.arm.set_joint_positions(
                        left_q_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )
                    right_bot.arm.set_joint_positions(
                        right_q_cmd.tolist(),
                        moving_time=cfg.moving_time,
                        accel_time=cfg.accel_time,
                        blocking=False,
                    )
                    t_cmd_elapsed = now() - t_cmd_start

                    cmd_kin.q_cmd[left_arm_indices] = left_q_cmd
                    cmd_kin.q_cmd[right_arm_indices] = right_q_cmd

                    fk_cmd = robot.forward_kinematics(cmd_kin.q_cmd)
                    cmd_kin.T_left_cmd = jaxlie.SE3(fk_cmd[left_ee_index]).as_matrix()
                    cmd_kin.T_right_cmd = jaxlie.SE3(fk_cmd[right_ee_index]).as_matrix()

                    cmd_state.last_left_cmd = left_q_cmd.copy()
                    cmd_state.last_right_cmd = right_q_cmd.copy()
                    cmd_state.last_arm_cmd_time = now()

                    TIMING_LOG["ik_solve"].append(t_solve_elapsed)
                    TIMING_LOG["cmd"].append(t_cmd_elapsed)

                    logging.info(
                        f"[TIMING] ik={t_solve_elapsed*1000:.2f} ms, cmd={t_cmd_elapsed*1000:.2f} ms"
                    )

        t_ik_elapsed = now() - t_ik_start

        # --- SECTION 4: read joint states + logging ---
        t_log_start = now()

        left_joint_state_msg = left_bot.dxl.joint_states
        right_joint_state_msg = right_bot.dxl.joint_states

        left_joint_positions = np.array(left_joint_state_msg.position[:6], dtype=float)
        right_joint_positions = np.array(right_joint_state_msg.position[:6], dtype=float)
        left_gripper_state = np.array([left_joint_state_msg.position[6]], dtype=float)
        right_gripper_state = np.array([right_joint_state_msg.position[6]], dtype=float)

        if teleop_state.active and recording:
            append_episode_step(
                episode_data,
                left_joint_positions,
                right_joint_positions,
                left_gripper_state,
                right_gripper_state,
                cmd_state,
                left_gripper_action,
                right_gripper_action,
            )

        t_log_elapsed = now() - t_log_start

        # --- SECTION 5: sleep / loop timing ---
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