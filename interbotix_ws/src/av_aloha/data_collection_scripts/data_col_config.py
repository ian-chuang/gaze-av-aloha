"""
Configuration for teleoperation and dataset collection.
"""

from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("/home/devi/giava/pyroki/examples")

import os

import numpy as np

DATASET_ROOT = ("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot")

TASKS = {
    1: "screwdriver_insertion",
    2: "block_square",
    3: "grasp_cube",
    4: "transfer_flower",
    5: "bimanual",
    6: "active_vision",
    7: "active_vision_data_collection",  # coupled-IK study deployment
}

# Defines which arms are active in each mode, and the corresponding action layout.

ARM_MODES = {
    "left": ["left"],
    "right": ["right"],
    "middle": ["middle"],
    "bimanual": ["left", "right"],
    "av": ["left", "right", "middle"],
}
# "all" and "av" are the same three-arm mode (move_arms.py says "all",
# data_collection says "av")
ARM_MODES["all"] = ARM_MODES["av"]

ACTION_LAYOUTS = {
    "left": {
        "left_arm": slice(0, 6),
        "left_gripper": 6,
    },

    "right": {
        "right_arm": slice(0, 6),
        "right_gripper": 6,
    },

    # The camera arm has 7 joints and no gripper -- build_action_names() lays
    # it out as 7 values, nothing after.  Missing until now, so replaying a
    # middle-only recording raised KeyError on a mode argparse accepted.
    "middle": {
        "middle_arm": slice(0, 7),
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

# ARM_MODES names the three-arm mode "av" while the layout above is keyed
# "all"; alias so replay_episode.py's ACTION_LAYOUTS[mode] works for
# av-mode datasets.
ACTION_LAYOUTS["av"] = ACTION_LAYOUTS["all"]

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
    # Control rate + scales, env-tunable (2026-08) so data_collection can be
    # tuned to match teleop_debug_tool behavior without code edits:
    #   GIAVA_CONTROL_HZ=20  GIAVA_POS_SCALE=1.0  GIAVA_CAM_POS_SCALE=0.6
    #   GIAVA_MOVING_TIME=0.14
    # Changing the rate also rescales the IK smoothing automatically
    # (CoupledStudyIK receives control_dt).
    control_dt: float = field(
        default_factory=lambda: 1.0 / float(os.environ.get("GIAVA_CONTROL_HZ", "50"))
    )
    position_scale: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_POS_SCALE", "1.35"))
    )
    alpha: float = 0.3
    arm_cmd_dt: float = field(
        default_factory=lambda: 1.0 / float(os.environ.get("GIAVA_CONTROL_HZ", "50"))
    )
    moving_time: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_MOVING_TIME", "0.14"))
    )
    accel_time: float = 0.04
    max_ee_step: float = 0.02
    pos_weight: float = 40.0
    ori_weight: float = 0.25
    dq_weight: float = 0.18
    joint_reached_tol: float = 0.03
    ee_reached_tol: float = 0.01
    cmd_timeout: float = 0.25
    full_joint_velocity_limits_value: float = 2.3

    # Post-solve per-joint step clamp.  DISABLED to match the ik_study
    # conditions: the study's winner was validated with no clamp, no LPF and no
    # reseed, and a saturating clamp creates the acceleration discontinuities it
    # is meant to prevent (the solver never learns its command was truncated).
    #
    # WARNING: nothing else bounds joint velocity.  The smoothing cost is a soft
    # penalty on deviation from the previous configuration -- it makes large
    # steps expensive, it does not make them impossible.  With this off, an IK
    # discontinuity goes straight to the motors.  Re-enable with
    # GIAVA_ENABLE_JOINT_CLAMP=1 if the arms move harder than you expect.
    enable_joint_clamp: bool = field(
        default_factory=lambda: os.environ.get("GIAVA_ENABLE_JOINT_CLAMP", "0") == "1"
    )

    # Driver-feasibility clamp (separate from the tuning clamp above, and ON by
    # default).  interbotix's `check_joint_limits` rejects the WHOLE group
    # command if any joint fails, so a single infeasible joint freezes the arm
    # completely -- it does not clip, it refuses.  Its test is
    #     |goal - last_command| / moving_time  >  joint_velocity_limit
    # so the largest command it will accept is  vl * moving_time  per tick.
    # Clamping to just under that keeps every command executable while leaving
    # far more headroom than the old tuning clamp (0.05-0.10 rad).
    enable_driver_clamp: bool = True
    # Headroom fraction of the driver-feasible step (env-tunable: this is the
    # safety knob to slow the arms down globally WITHOUT the old per-joint
    # tuning clamp and its saturate-release jerk).
    #   0.9 -> max ~2.8 rad/s equivalent; 0.45 -> ~1.4 rad/s.
    driver_velocity_limit: float = 3.141593  # rad/s, from group_info
    driver_clamp_safety: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_DRIVER_CLAMP_SAFETY", "0.9"))
    )
    driver_limit_margin: float = 0.03        # rad cushion inside each DRIVER
    # position limit.  0.005 proved too thin: commands landing exactly on the
    # limit get rejected outright (right wrist_angle at +2.243 vs +2.234), and a
    # rejected command freezes the driver's reference.  ~1.7 deg of range given
    # up per joint end buys never hitting the wall.

    @property
    def driver_max_step(self) -> float:
        """Largest per-tick joint delta the driver will accept."""
        return self.driver_clamp_safety * self.driver_velocity_limit * self.moving_time

    # Per-instance NumPy arrays via default_factory
    max_joint_step: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.05, 0.06, 0.08, 0.08, 0.10],
            dtype=float,
        )
    )
    # 7-dof middle arm (wx250s_7dof: extra wrist joint); same per-joint
    # progression with a wrist-class limit for the 7th
    max_joint_step_middle: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.05, 0.06, 0.08, 0.08, 0.10, 0.10],
            dtype=float,
        )
    )
    R_arm_remap: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=float,
        )
    )
    # Head pose (HPosition/HRotation) arrives from the SAME headset runtime and
    # world frame as the hand controllers, so the camera arm needs the SAME
    # remap as the gripper arms.  This was identity (i.e. never filled in),
    # which fed raw headset axes to the camera target: lateral head motion
    # mapped to robot +y (operator-backward) instead of +x (operator-left), and
    # head yaw (about headset up) was applied about the robot's backward axis --
    # producing the observed pitch/yaw swaps.  With the shared remap, the
    # world-frame rotation composition (R @ dR @ R^T) sends head yaw -> robot
    # yaw (about +z), pitch -> pitch (about x), roll -> roll (about y).
    # Remap for the middle (head) target.  The head pose is converted from its
    # native Unity-style basis into the CONTROLLER convention at parse time
    # (HEAD_BASIS_FIX in data_collection.py), so the same remap as the gripper
    # arms applies here.  Net head->robot mapping is identical to the previous
    # dedicated matrix (M @ C == B, verified); doing the conversion once at the
    # source also fixes head motion leaking into the gripper arms through the
    # head-relative hand composition.
    R_cam_remap: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=float,
        )
    )

    # Camera-arm sensitivity.  The head is never still -- breathing and weight
    # shifts move it by millimetres continuously, and at position_scale 1.0 the
    # camera arm chases all of it.  The deadband zeroes deltas below ~1.5 cm
    # (soft-edged, so there is no snap at the threshold) and the separate scale
    # lets head motion map sub-unity to arm motion.
    cam_position_scale: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_CAM_POS_SCALE", "0.6"))
    )
    cam_deadband_m: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_CAM_DEADBAND_M", "0.015"))
    )
    # Camera-arm step-clamp window.  The shared max_ee_step (0.02) clamps the
    # target to within 2 cm of the arm's CURRENT pose; at position weight 10
    # that caps the IK's position cost at (10*0.02)^2 = 0.04 -- a whisper next
    # to the orientation terms, so the position channel starves (x/y appear
    # dead).  A 5 cm window gives the position cost a signal worth acting on
    # while keeping the anti-windup property.  See TELEOP_MATH.md section 4.
    cam_max_ee_step: float = field(
        default_factory=lambda: float(os.environ.get("GIAVA_CAM_MAX_EE_STEP", "0.05"))
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
    # Session-calibrated remap (raw app world -> robot), built at anchor time
    # from the head's horizontal gaze.  The probe (2026-08) measured the app's
    # world frame as z-up but with an ARBITRARY per-session yaw (+43 deg that
    # run) -- set by where the headset faced at app start -- so a fixed remap
    # matrix is wrong by a different angle every session.  None = fall back to
    # the static matrix (pre-calibration behavior).
    session_remap: Optional[np.ndarray] = None

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
## session_yaw_remap and HEAD_LOCAL_FWD live in transform_utils (the shared
## frame-math helper) and are re-exported here for the teleop call sites.
try:
    from .transform_utils import HEAD_LOCAL_FWD, session_yaw_remap  # noqa: F401
except ImportError:
    from transform_utils import HEAD_LOCAL_FWD, session_yaw_remap  # noqa: F401


def anchor_arm_state(state_arm, controller_pose, cmd_pose, head_pose=None, base_remap=None):
    """(Re)anchor one arm's teleop reference frames to NOW.

    Called at session start for every arm, and again whenever an individual
    arm's button is pressed mid-session: with per-arm buttons (Y = middle),
    an arm can become active long after the session started, and its anchor
    from session start would be stale -- the arm would jump by however far
    the controller/head moved in between."""
    quat_wxyz = cmd_pose[:4]
    pos = cmd_pose[4:]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw).as_matrix()
    state_arm.start_controller_pos = controller_pose[:3, 3].copy()
    state_arm.start_controller_rot = controller_pose[:3, :3].copy()
    state_arm.start_robot_pos = np.asarray(pos).copy()
    state_arm.start_robot_rot = rot.copy()
    state_arm.filtered_target_pos = np.asarray(pos).copy()
    if head_pose is not None and base_remap is not None:
        remap = session_yaw_remap(head_pose, base_remap)
        if remap is not None:
            state_arm.session_remap = remap
        elif state_arm.session_remap is None:
            print("[frame] head gaze too vertical at anchor -- using static remap")


def start_teleop_session(state, mode, controller_poses, cmd_kin, cfg=None):
    state.active = True

    head_pose = controller_poses.get("middle")  # the raw head pose
    for arm in ARM_MODES[mode]:
        state.arms[arm] = ArmTeleopState()
        base = None
        if cfg is not None:
            base = cfg.R_cam_remap if arm == "middle" else cfg.R_arm_remap
        anchor_arm_state(state.arms[arm], controller_poses[arm], cmd_kin.T_cmd[arm],
                         head_pose=head_pose, base_remap=base)

# Computes the target end-effector position and orientation for a given arm based on the current controller pose, the initial poses, and the configuration parameters.
def compute_gripper_arm_target(cfg, arm_state, controller_pose, current_pose, remap_matrix):
    if arm_state.session_remap is not None:
        remap_matrix = arm_state.session_remap
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
    if arm_state.session_remap is not None:
        remap_matrix = arm_state.session_remap
    delta_ctrl = controller_pose[:3, 3] - arm_state.start_controller_pos
    delta_robot = remap_matrix @ delta_ctrl
    # Soft deadband: ignore the head's constant millimetre-level wander without
    # a snap when crossing the threshold (magnitude shrinks by the deadband).
    mag = float(np.linalg.norm(delta_robot))
    if mag <= cfg.cam_deadband_m:
        delta_robot = np.zeros(3)
    else:
        delta_robot = delta_robot * ((mag - cfg.cam_deadband_m) / mag)
    raw_target = arm_state.start_robot_pos + cfg.cam_position_scale * delta_robot
    arm_state.filtered_target_pos = cfg.alpha * raw_target + (1.0 - cfg.alpha) * arm_state.filtered_target_pos
    target_pos = clamp_cartesian_step(arm_state.filtered_target_pos, np.asarray(current_pose[4:], dtype=float), cfg.cam_max_ee_step)

    # Head rotation delta composed in the WORLD frame (delta @ start), not
    # the EE-local frame (start @ delta): local-frame application rotates
    # about the camera link's own axes, which do NOT line up with the
    # world's — at the forward pose the link z is horizontal, so head yaw
    # became camera pitch.  World-frame composition keeps yaw = yaw and
    # pitch = pitch regardless of the link's local frame convention.
    R_delta_world = controller_pose[:3, :3] @ arm_state.start_controller_rot.T
    R_delta_robot = remap_matrix @ R_delta_world @ remap_matrix.T
    target_rot = R_delta_robot @ arm_state.start_robot_rot
    target_wxyz = quat_xyzw_to_wxyz(R.from_matrix(target_rot).as_quat())

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
    import pyroki_snippets as pks

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
