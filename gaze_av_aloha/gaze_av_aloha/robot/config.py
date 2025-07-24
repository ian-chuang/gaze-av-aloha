import pathlib
import os
# task parameters
# XML_DIR is assets/aloha_real.xml
XML_DIR = os.path.join(str(pathlib.Path(__file__).parent.resolve()), 'assets')

# control parameters
REAL_DT = 0.03
FPS = round(1 / REAL_DT)

# physics parameters
SIM_PHYSICS_DT=0.002
SIM_DT = 0.04
SIM_PHYSICS_ENV_STEP_RATIO = int(SIM_DT/SIM_PHYSICS_DT)
SIM_DT = SIM_PHYSICS_DT * SIM_PHYSICS_ENV_STEP_RATIO

# robot parameters
LEFT_ARM_POSE = [0, -0.082, 1.06, 0, -0.953, 0, 0.02239]
RIGHT_ARM_POSE = [0, -0.082, 1.06, 0, -0.953, 0, 0.02239]
MIDDLE_ARM_POSE = [0, -0.8, 0.8, 0, 0.5, 0, 0]
LEFT_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
RIGHT_JOINT_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]
MIDDLE_JOINT_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
    "middle_wrist_3_joint",
]
LEFT_ACTUATOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
RIGHT_ACTUATOR_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
]
MIDDLE_ACTUATOR_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
    "middle_wrist_3_joint",
]
LEFT_EEF_SITE = "left_gripper_control"
RIGHT_EEF_SITE = "right_gripper_control"
MIDDLE_EEF_SITE = "middle_zed_camera_center"

# Gripper joint limits (qpos[6])
LEFT_GRIPPER_JOINT_OPEN = 0.05522330850362778
LEFT_GRIPPER_JOINT_CLOSE = -1.006291389465332
RIGHT_GRIPPER_JOINT_OPEN = -0.6626797318458557
RIGHT_GRIPPER_JOINT_CLOSE = -1.682776927947998

############################ Helper functions ############################
LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - LEFT_GRIPPER_JOINT_CLOSE) / (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE)
LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE) + LEFT_GRIPPER_JOINT_CLOSE
RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - RIGHT_GRIPPER_JOINT_CLOSE) / (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE)
RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE) + RIGHT_GRIPPER_JOINT_CLOSE
LEFT_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE)
RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE)