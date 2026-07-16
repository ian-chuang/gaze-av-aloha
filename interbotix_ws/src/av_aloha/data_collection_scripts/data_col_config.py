"""
Configuration for teleoperation and dataset collection.
"""

from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki as pk
import pyroki_snippets as pks
from transform_utils import (
    quat_xyzw_to_wxyz,
)

import numpy as np

DATASET_ROOT = ("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot")

TASKS = {
    1: "screwdriver_insertion",
    2: "block_square",
    3: "grasp_cube",
    4: "transfer_flower",
    5: "bimanual_data_collection",
}

# Defines which arms are active in each mode, and the corresponding action layout.

ARM_MODES = {
    "left": ["left"],
    "right": ["right"],
    "middle": ["middle"],
    "bimanual": ["left", "right"],
    "all": ["left", "right", "middle"],
}

ACTION_LAYOUTS = {
    "left": {
        "left_arm": slice(0, 6),
        "left_gripper": 6,
    },

    "right": {
        "right_arm": slice(0, 6),
        "right_gripper": 6,
    },

    "bimanual": {
        "left_arm": slice(0, 6),
        "left_gripper": 6,

        "right_arm": slice(7, 13),
        "right_gripper": 13,
    },

    "all": {
        "left_arm": slice(0, 6),
        "left_gripper": 6,

        "right_arm": slice(7, 13),
        "right_gripper": 13,

        "middle_arm": slice(14, 21),
    },
}

def clamp_joint_step(q_curr, q_target, max_step):
    dq = np.clip(q_target - q_curr, -max_step, max_step)
    return q_curr + dq

def clamp_cartesian_step(target, prev_target, max_step):
    delta = target - prev_target
    norm = np.linalg.norm(delta)
    if norm > max_step and norm > 1e-9:
        delta *= (max_step / norm)
    return prev_target + delta

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
    R_arm_remap: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=float,
        )
    )
    R_cam_remap: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=float,
        )
    )

# State for each arm during teleoperation, tracking the initial controller and robot poses, as well as a filtered target position for smooth motion.
@dataclass
class ArmTeleopState:
    active: bool = False
    start_controller_pos: Optional[np.ndarray] = None
    start_controller_rot: Optional[np.ndarray] = None
    start_robot_pos: Optional[np.ndarray] = None
    start_robot_rot: Optional[np.ndarray] = None
    filtered_target_pos: Optional[np.ndarray] = None

# Overall teleoperation session state, including whether it's active and the state for each arm.
@dataclass
class TeleopSessionState:
    active: bool = False
    arms: dict[str, ArmTeleopState] = field(default_factory=dict)

# State for tracking the last command times and values for each arm, used to implement command timeouts and ensure smooth control.
@dataclass
class RobotCommandState:
    last_arm_cmd_time: float = 0.0
    last_cmds: dict[str, np.ndarray] = field(default_factory=dict)

# Kinematics state for the commanded target poses of each arm, used to compute the desired end-effector positions and orientations based on the controller input and initial poses.
@dataclass
class CommandKinematicsState:
    q_cmd: Optional[np.ndarray] = None
    T_cmd: dict[str, np.ndarray] = field(default_factory=dict)

# Initializes the teleoperation session state for the active arms based on the current controller poses and commanded kinematics.
def start_teleop_session(state, mode, controller_poses, cmd_kin):
    state.active = True

    for arm in ARM_MODES[mode]:

        pose = cmd_kin.T_cmd[arm]

        quat_wxyz = pose[:4]
        pos = pose[4:]

        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ])

        rot = R.from_quat(quat_xyzw).as_matrix()

        state.arms[arm] = ArmTeleopState(
            start_controller_pos=controller_poses[arm][:3, 3].copy(),
            start_controller_rot=controller_poses[arm][:3, :3].copy(),

            start_robot_pos=np.asarray(pos).copy(),
            start_robot_rot=rot.copy(),

            filtered_target_pos=np.asarray(pos).copy(),
        )

# Computes the target end-effector position and orientation for a given arm based on the current controller pose, the initial poses, and the configuration parameters.
def compute_gripper_arm_target(cfg, arm_state, controller_pose, current_pose, remap_matrix):
    delta_ctrl = controller_pose[:3, 3] - arm_state.start_controller_pos
    delta_robot = remap_matrix @ delta_ctrl
    raw_target = arm_state.start_robot_pos + cfg.position_scale * delta_robot
    arm_state.filtered_target_pos = cfg.alpha * raw_target + (1.0 - cfg.alpha) * arm_state.filtered_target_pos
    target_pos = clamp_cartesian_step(arm_state.filtered_target_pos, np.asarray(current_pose[4:], dtype=float), cfg.max_ee_step)

    R_delta = arm_state.start_controller_rot.T @ controller_pose[:3,:3]
    rotvec = R.from_matrix(R_delta).as_rotvec()
    pitch = rotvec[1]
    yaw = rotvec[0]
    roll = rotvec[2]
    rotvec_ee = np.array([-pitch, yaw, roll])
    target_rot = arm_state.start_robot_rot @ R.from_rotvec(rotvec_ee).as_matrix()
    target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(target_rot).as_quat())

    return target_pos, target_wxyz

def compute_camera_arm_target(cfg, arm_state, controller_pose, current_pose, remap_matrix):
    delta_ctrl = controller_pose[:3, 3] - arm_state.start_controller_pos
    delta_robot = remap_matrix @ delta_ctrl
    raw_target = arm_state.start_robot_pos + cfg.position_scale * delta_robot
    arm_state.filtered_target_pos = cfg.alpha * raw_target + (1.0 - cfg.alpha) * arm_state.filtered_target_pos
    target_pos = clamp_cartesian_step(arm_state.filtered_target_pos, np.asarray(current_pose[4:], dtype=float), cfg.max_ee_step)

    R_delta = arm_state.start_controller_rot.T @ controller_pose[:3,:3]
    target_rot = arm_state.start_robot_rot @ R_delta
    target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(target_rot).as_quat())

    # print("\n\ndelta_ctrl", delta_ctrl)
    # print("delta_robot", delta_robot)

    return target_pos, target_wxyz

def stop_teleop_session(state):
    state.active = False

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