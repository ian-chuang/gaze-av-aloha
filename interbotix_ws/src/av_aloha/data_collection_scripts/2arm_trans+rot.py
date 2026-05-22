import rospy
import time
import numpy as np

from scipy.spatial.transform import Rotation as R

from interbotix_xs_modules.arm import (
    InterbotixManipulatorXS
)

from interbotix_xs_msgs.msg import (
    JointSingleCommand
)

import pyroki as pk
from yourdfpy import URDF
import jaxlie

from webrtc_headset import WebRTCHeadset
from transform_utils import pose2mat

import sys
sys.path.append(
    "/home/devi/giava/pyroki/examples"
)

import pyroki_snippets as pks

import depthai as dai
import time
import cv2
import numpy as np
from webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback
from transform_utils import (
    pose2mat,
)


# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "leftgripper_base"
RIGHT_EE_LINK = "rightgripper_base"

CONTROL_DT = 0.05

MOVING_TIME = 0.07
ACCEL_TIME = 0.02

POSITION_SCALE = 1.0

ALPHA = 0.2

# ============================================================
# HELPERS
# ============================================================

def quat_xyzw_to_wxyz(q):

    return np.array([
        q[3],
        q[0],
        q[1],
        q[2],
    ])



# ============================================================
# MAIN
# ============================================================

def main():

    rospy.init_node(
        "bimanual_vr_teleop"
    )

    # ========================================================
    # HEADSET
    # ========================================================

    headset = WebRTCHeadset()

    headset.run_in_thread()

    # CAMERA

    # ---- Setup headset ----
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()

    # ---- Setup pipeline ----
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_out = cam_left.requestOutput(
        (1280, 800),
        type=dai.ImgFrame.Type.GRAY8,
        fps=60,
    )

    right_out = cam_right.requestOutput(
        (1280, 800),
        type=dai.ImgFrame.Type.GRAY8,
        fps=60,
    )

    q_left = left_out.createOutputQueue()
    q_right = right_out.createOutputQueue()

    pipeline.start()

    time.sleep(0.5)



    # ========================================================
    # ROBOT MODEL
    # ========================================================

    urdf = URDF.load(
        URDF_PATH
    )

    robot = pk.Robot.from_urdf(
        urdf
    )

    # for name in robot.joints.actuated_names:

    #     if "middle" in name:

    #         print(name)

    # ========================================================
    # JOINT GROUPS
    # ========================================================

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

    # MIDDLE_ARM_NAMES = [
    #     "waist",
    #     "shoulder",
    #     "elbow",
    #     "forearm_roll",
    #     "wrist_angle",
    #     "camera_roll",
    # ]

    MIDDLE_ARM_NAMES = [
        "middlebase_link",
        "middleshoulder_link",
        "middleupper_arm_link",
        "middleupper_forearm_link",
        "middlelower_forearm_link",
        "middlewrist_link",
        "middlepan_link",
    ]

    left_arm_indices = [

        robot.joints.actuated_names.index(name)

        for name in LEFT_ARM_NAMES
    ]

    right_arm_indices = [

        robot.joints.actuated_names.index(name)

        for name in RIGHT_ARM_NAMES
    ]

    middle_arm_indices = [

        robot.joints.actuated_names.index(name)

        for name in MIDDLE_ARM_NAMES
    ]

    # ========================================================
    # EE INDICES
    # ========================================================

    left_ee_index = robot.links.names.index(
        LEFT_EE_LINK
    )

    right_ee_index = robot.links.names.index(
        RIGHT_EE_LINK
    )

    # GRIPPER
    left_gripper_command = JointSingleCommand(
        name="gripper"
    )

    right_gripper_command = JointSingleCommand(
        name="gripper"
    )

    LEFT_GRIPPER_OPEN = 1.5
    LEFT_GRIPPER_CLOSED = -1.5

    RIGHT_GRIPPER_OPEN = 1.5
    RIGHT_GRIPPER_CLOSED = -1.5

    # ========================================================
    # INTERBOTIX
    # ========================================================

    left_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    left_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    left_bot.dxl.robot_torque_enable("single", "gripper", True)
    #left_bot.dxl.robot_set_operating_modes("group", "arm", "position")

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)
    #right_bot.dxl.robot_set_operating_modes("group", "arm", "position")

    middle_bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="puppet_middle",
        moving_time=2.0,
        accel_time=0.5,
        init_node=False,
    )

    middle_bot.dxl.robot_set_operating_modes(
        "group",
        "arm",
        "position",
    )

    middle_bot.dxl.robot_torque_enable(
        "group",
        "arm",
        True,
    )

    left_q = [0.0, -1.8, 1.6, 0.0, 0.6, 0.0, 0.0]

    right_q = [0.0, -1.8, 1.6, 0.0, 0.6, 0.0, 0.0]

    middle_q = [0.05, -1.5, 0.05, -0.08, 2.0, 1.4, 0.0]

    # middle_bot.arm.set_joint_positions(
    #     middle_q.tolist(),
    #     moving_time=2.0,
    #     blocking=False,
    # )

    left_bot.arm.set_joint_positions(
        left_q[:6],
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    right_bot.arm.set_joint_positions(
        right_q[:6],
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    middle_bot.arm.set_joint_positions(
        middle_q[:6],
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    time.sleep(2)
    rospy.sleep(2.0)

    # print(
    #     middle_bot.arm.group_info.joint_names
    # )

    # print(
    #     len(
    #         middle_bot.arm.group_info.joint_names
    #     )
    # )

    rospy.sleep(1.0)

    # ========================================================
    # FULL ROBOT STATE
    # ========================================================

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    q[left_arm_indices] = np.array(
        left_bot.arm.get_joint_commands()
    )

    q[right_arm_indices] = np.array(
        right_bot.arm.get_joint_commands()
    )

    q[middle_arm_indices] = np.array(
        middle_q
    )


    # ========================================================
    # FK
    # ========================================================

    fk = robot.forward_kinematics(q)

    T_left = jaxlie.SE3(
        fk[left_ee_index]
    ).as_matrix()

    T_right = jaxlie.SE3(
        fk[right_ee_index]
    ).as_matrix()

    # ========================================================
    # INITIAL TARGETS
    # ========================================================

    left_start_robot_position = (
        T_left[:3,3].copy()
    )

    right_start_robot_position = (
        T_right[:3,3].copy()
    )

    left_filtered_target_position = (
        left_start_robot_position.copy()
    )

    right_filtered_target_position = (
        right_start_robot_position.copy()
    )

    # ========================================================
    # FIXED ORIENTATION
    # ========================================================

    left_fixed_rot = R.from_euler(
        "xyz",
        [-90, 0, -90],
        degrees=True,
    )

    right_fixed_rot = R.from_euler(
        "xyz",
        [-90, 0, 90],
        degrees=True,
    )

    left_target_wxyz = quat_xyzw_to_wxyz(
        left_fixed_rot.as_quat()
    )

    right_target_wxyz = quat_xyzw_to_wxyz(
        right_fixed_rot.as_quat()
    )

    # ========================================================
    # VELOCITY LIMITS
    # ========================================================

    full_joint_velocity_limits = np.ones(
        robot.joints.num_actuated_joints
    ) * 3.0

    # ========================================================
    # REMAP
    # ========================================================

    R_remap_left = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])

    R_remap_right = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])

    # ========================================================
    # TELEOP STATE
    # ========================================================

    teleop_active = False

    left_start_controller_position = None
    right_start_controller_position = None

    left_start_controller_rot = None
    right_start_controller_rot = None

    left_start_robot_rot = None
    right_start_robot_rot = None

    # ========================================================
    # LOOP
    # ========================================================

    print("\\nREADY")

    while not rospy.is_shutdown():

        headset_data = (
            headset.receive_data()
        )

        if headset_data is None:

            time.sleep(0.01)

            continue

        # ----------------------------------------------------
        # CONTROLLERS
        # ----------------------------------------------------

        left_controller = pose2mat(
            headset_data.l_pos,
            headset_data.l_quat,
        )

        right_controller = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        button_pressed = (
            headset_data.r_button_one
        )

        left_trigger = headset_data.l_index_trigger
        right_trigger = headset_data.r_index_trigger

        """ 
        Debugging controller triggers (left/right, index/hand)
        
        left_trigger = headset_data.l_index_trigger
        if left_trigger > 0:
            print("LEFT TRIGGER:", left_trigger)
        right_trigger = headset_data.r_index_trigger
        if right_trigger > 0:
            print("RIGHT TRIGGER:", right_trigger)

        left_trigger2 = headset_data.l_hand_trigger
        if left_trigger2 > 0:
            print("LEFT HAND TRIGGER:", left_trigger2)
        right_trigger2 = headset_data.r_hand_trigger
        if right_trigger2 > 0:
            print("RIGHT HAND TRIGGER:", right_trigger2) """

        # ====================================================
        # ENABLE TELEOP
        # ====================================================

        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            left_start_controller_position = (
                left_controller[:3,3].copy()
            )

            right_start_controller_position = (
                right_controller[:3,3].copy()
            )

            left_start_robot_position = (
                T_left[:3,3].copy()
            )

            right_start_robot_position = (
                T_right[:3,3].copy()
            )

            left_start_controller_rot = left_controller[:3,:3].copy()
            right_start_controller_rot = right_controller[:3,:3].copy()

            left_start_robot_rot = T_left[:3,:3].copy()
            right_start_robot_rot = T_right[:3,:3].copy()

            print("\\nTeleop ENABLED")

        # ====================================================
        # DISABLE TELEOP
        # ====================================================

        elif (
            not button_pressed
            and teleop_active
        ):

            teleop_active = False

            print("\\nTeleop DISABLED")

        # ====================================================
        # TELEOP
        # ====================================================

        if teleop_active:

            # SEND IMAGES TO HEADSET
            left_img = q_left.get().getCvFrame()
            right_img = q_right.get().getCvFrame()

            headset.send_images(left_img, right_img)

            

            # ------------------------------------------------
            # LEFT DELTA
            # ------------------------------------------------

            left_delta_controller = (

                left_controller[:3,3]

                - left_start_controller_position
            )

            left_delta_robot = (
                R_remap_left
                @ left_delta_controller
            )

            left_raw_target_position = (

                left_start_robot_position

                + POSITION_SCALE
                * left_delta_robot
            )

            left_filtered_target_position = (

                ALPHA
                * left_raw_target_position

                + (1 - ALPHA)
                * left_filtered_target_position
            )

            left_target_position = (
                left_filtered_target_position
            )

            # ------------------------------------------------
            # RIGHT DELTA
            # ------------------------------------------------

            right_delta_controller = (

                right_controller[:3,3]

                - right_start_controller_position
            )

            right_delta_robot = (
                R_remap_right
                @ right_delta_controller
            )

            right_raw_target_position = (

                right_start_robot_position

                + POSITION_SCALE
                * right_delta_robot
            )

            right_filtered_target_position = (

                ALPHA
                * right_raw_target_position

                + (1 - ALPHA)
                * right_filtered_target_position
            )

            right_target_position = (
                right_filtered_target_position
            )

            # =================================================
            # GRIPPERS
            # =================================================

            cmd = JointSingleCommand(
                name="gripper"
            )

            if right_trigger > 0:
                cmd.cmd = -1.7
                right_bot.gripper.core.pub_single.publish(cmd)
            else:
                cmd.cmd = 0.1
                right_bot.gripper.core.pub_single.publish(cmd)

            if left_trigger > 0:
                cmd.cmd = -1.7
                left_bot.gripper.core.pub_single.publish(cmd)
            else:
                cmd.cmd = 0.1
                left_bot.gripper.core.pub_single.publish(cmd)

            # =================================================
            # ROTATION
            # =================================================

            left_current_rot = left_controller[:3,:3]
            right_current_rot = right_controller[:3,:3]

            left_delta_rot = (
                left_current_rot
                @ left_start_controller_rot.T
            )

            # left_euler = R.from_matrix(
            #     left_delta_rot
            # ).as_euler("xyz")

            # # invert pitch
            # #left_euler[1] *= -1

            # left_delta_rot = R.from_euler(
            #     "xyz",
            #     left_euler
            # ).as_matrix()

            right_delta_rot = (
                right_current_rot
                @ right_start_controller_rot.T
            )

            # left_target_rot = (
            #     left_delta_rot
            #     @ left_start_robot_rot
            # )

            # left_delta_rot_robot = (
            #     R_remap_left
            #     @ left_delta_rot
            #     @ R_remap_left.T
            # )

            left_target_rot = (
                left_delta_rot
                @ left_start_robot_rot
            )

            right_target_rot = (
                right_delta_rot
                @ right_start_robot_rot
            )

            left_target_wxyz = quat_xyzw_to_wxyz(
                R.from_matrix(left_target_rot).as_quat()
            )

            right_target_wxyz = quat_xyzw_to_wxyz(
                R.from_matrix(right_target_rot).as_quat()
            )

            # print(vars(headset_data))

            # =================================================
            # WHOLE BODY IK
            # =================================================

            q_new = pks.solve_trajectories_ik(

            robot=robot,

            target_link_names=[
                LEFT_EE_LINK,
                RIGHT_EE_LINK,
            ],

            target_positions=[
                left_target_position,
                right_target_position,
            ],

            target_wxyzs=[
                left_target_wxyz,
                right_target_wxyz,
            ],

            prev_q=q,

            # q_nominal=q_nominal,

            # nominal_weight=0.5,

            dt=CONTROL_DT,

            joint_velocity_limits=(
                full_joint_velocity_limits
            ),
        )

            # =================================================
            # VALID IK
            # =================================================

            if q_new is not None:

                q = q_new

                # --------------------------------------------
                # LEFT
                # --------------------------------------------

                left_q = q[
                    left_arm_indices
                ]

                left_bot.arm.set_joint_positions(

                    left_q.tolist(),

                    moving_time=MOVING_TIME,

                    accel_time=ACCEL_TIME,

                    blocking=False,
                )

                # --------------------------------------------
                # RIGHT
                # --------------------------------------------

                right_q = q[
                    right_arm_indices
                ]

                right_bot.arm.set_joint_positions(

                    right_q.tolist(),

                    moving_time=MOVING_TIME,

                    accel_time=ACCEL_TIME,

                    blocking=False,
                )

                # --------------------------------------------
                # FK UPDATE
                # --------------------------------------------

                fk = robot.forward_kinematics(q)

                T_left = jaxlie.SE3(
                    fk[left_ee_index]
                ).as_matrix()

                T_right = jaxlie.SE3(
                    fk[right_ee_index]
                ).as_matrix()

            else:

                print("IK FAILED")

        time.sleep(CONTROL_DT)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()