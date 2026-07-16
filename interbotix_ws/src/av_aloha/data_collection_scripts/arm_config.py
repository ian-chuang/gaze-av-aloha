"""
Robot definitions, joint names, and predefined poses.
"""

import numpy as np

URDF_PATH = "/home/devi/giava/giava.urdf"

RIGHT_EE_LINK = "rightgripper_base"
LEFT_EE_LINK = "leftgripper_base"
MIDDLE_EE_LINK = "middlepan_link"

LEFT_ARM_JOINT_NAMES = ["leftwaist", "leftshoulder", "leftelbow", "leftforearm_roll", "leftwrist_angle", "leftwrist_rotate"]
RIGHT_ARM_JOINT_NAMES = ["rightwaist", "rightshoulder", "rightelbow", "rightforearm_roll", "rightwrist_angle", "rightwrist_rotate"]
MIDDLE_ARM_JOINT_NAMES = ['middlebase', 'middleshoulder', 'middleupper_arm', 'middleupper_forearm', 'middlelower_forearm', 'middlewrist', 'middlepan']
# MIDDLE_ARM_JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'camera_roll']

# Configuration for each arm, including robot name, model, joint names, and end-effector link.

ARM_CONFIG = {
    "left": {
        "robot_name": "puppet_left",
        "robot_model": "vx300s",
        "has_gripper": True,
        "num_joints": 6,
        "joint_names": LEFT_ARM_JOINT_NAMES,
        "ee_link": LEFT_EE_LINK,
    },

    "right": {
        "robot_name": "puppet_right",
        "robot_model": "vx300s",
        "has_gripper": True,
        "num_joints": 6,
        "joint_names": RIGHT_ARM_JOINT_NAMES,
        "ee_link": RIGHT_EE_LINK,
    },

    "middle": {
        "robot_name": "puppet_middle",
        "robot_model": "wx250s",
        "has_gripper": False,
        "num_joints": 7,
        "joint_names": MIDDLE_ARM_JOINT_NAMES,
        "ee_link": MIDDLE_EE_LINK,
    },
}

# Predefined joint configurations for useful poses.

HIGH = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float)
LOW = np.array([0.02, 0.037, 0.598, -0.143, 0.986, 0.038], dtype=float)
FORWARD = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=float)
REST = np.array([0.0, -1.9, 1.635, 0.0, 0.7, 0.0], dtype=float)


DEFAULT_RESET_POSE = "forward"

M_HIGH = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 1.5, 0.0], dtype=float)
M_LOW = np.array([0.02, 0.037, 0.598, -0.143, 0.986, 1.5, 0.0], dtype=float)
M_FORWARD = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 1.5, 0.5], dtype=float)
M_REST = np.array([0.0, -1.9, 1.635, 0.0, 0.7, 1.5, 0.0], dtype=float)

POSES = {
    "left": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW},
    "right": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW},
    "middle": {"high": M_HIGH, "forward": M_FORWARD, "rest": M_REST, "low": M_LOW},
}