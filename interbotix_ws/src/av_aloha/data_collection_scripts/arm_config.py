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

M_HIGH = np.array([-3.1185829639434814, -0.13345633447170258, -0.7240389585494995, 0.0, 2.112116756439209, 1.6428934335708618, 2.3], dtype=float)
M_LOW = np.array([-3.130854845046997, -1.3959225416183472, 1.087592363357544, -0.05675728991627693, 0.6856894493103027, 1.6444274187088013, 2.3], dtype=float)
M_FORWARD = np.array([-3.0986413955688477, -1.310019612312317, 0.8805050253868103, -0.07516506314277649, 0.5307573676109314, 1.6428934335708618, 2.3], dtype=float)
M_REST = np.array([-3.1323888301849365, -1.8944662809371948, 1.5938060283660889, -0.09357283264398575, 0.725572943687439, 1.5800002813339233, 2.3], dtype=float)

M_FAR_SCENE = np.array([-3.15079665184021, -1.4695535898208618, -0.49087387323379517, -0.01840776950120926, 2.112116756439209, 1.7241944074630737, 2.3], dtype=float)
M_LOOKING_LEFT = np.array([-4.178563594818115, 0.6366020441055298, 0.04908738657832146, 1.3054176568984985, 1.8545827865600586, -0.6427379846572876, 2.3], dtype=float)
M_LOOKING_RIGHT = np.array([-2.113825559616089, 0.771592378616333, -0.31139811873435974, 1.7272623777389526, -1.7717478275299072, 0.5629709362983704, 2.3], dtype=float)

POSES = {
    "left": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW},
    "right": {"high": HIGH, "forward": FORWARD, "rest": REST, "low": LOW},
    "middle": {"high": M_HIGH, "forward": M_FORWARD, "rest": M_REST, "low": M_LOW, "far_scene": M_FAR_SCENE, "looking_left": M_LOOKING_LEFT, "looking_right": M_LOOKING_RIGHT},
}

"""
------------------
MIDDLE ARM POSES
------------------

REST: position: [-3.1323888301849365, -1.8944662809371948, 1.5938060283660889, -0.09357283264398575, 0.725572943687439, 1.5800002813339233, -3.9131851196289062]

FORWARD: position: [-3.0986413955688477, -1.310019612312317, 0.8805050253868103, -0.07516506314277649, 0.5307573676109314, 1.6428934335708618, 2.3]

LOW: position: [-3.130854845046997, -1.3959225416183472, 1.087592363357544, -0.05675728991627693, 0.6856894493103027, 1.6444274187088013, 2.3]

HIGH: position: [-3.1185829639434814, -0.13345633447170258, -0.7240389585494995, 0.0, 2.172116756439209, 1.6428934335708618, 2.3]

FAR SCENE: position: [-3.15079665184021, -1.4695535898208618, -0.49087387323379517, -0.01840776950120926, 2.172116756439209, 1.7241944074630737, 2.3]

LOOKING LEFT: position: [-4.178563594818115, 0.6366020441055298, 0.04908738657832146, 1.3054176568984985, 1.8545827865600586, -0.6427379846572876, 2.3]

LOOKING RIGHT: position: [-2.113825559616089, 0.771592378616333, -0.31139811873435974, 1.7272623777389526, -1.7717478275299072, 0.5629709362983704, 2.3]

-------------------------
MIDDLE ARM JOINT LIMITS
-------------------------

  - waist:        -6.35, 0.00
  - shoulder:     -1.89, 1.70
  - elbow:        -2.13, 1.58
  - forearm_roll: -1.58, 1.51
  - wrist_angle:  -1.54, 2.12
  - camera_roll:  -2.78, 2.99
  - camera_yaw:   -3.10, 3.07        (1.00 = looking right)


"""