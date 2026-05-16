import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import pyroki as pk
from yourdfpy import URDF

from interbotix_xs_modules.arm import InterbotixManipulatorXS

from webrtc_headset import WebRTCHeadset

from transform_utils import (
    pose2mat,
    align_rotation_to_z_axis,
)

import sys
sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks


# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "leftgripper_base"

CONTROL_DT = 0.03

MOVING_TIME = 0.08
ACCEL_TIME = 0.02

ALPHA = 0.25

COMMAND_DEADBAND = 0.005

# CONTROL_DT = 0.10

POSITION_SCALE = 0.3

# # joint low-pass filtering
# ALPHA = 0.1

# # interbotix timing
# MOVING_TIME = 0.4
# ACCEL_TIME = 0.1

# # minimum joint change before sending command
# COMMAND_DEADBAND = 0.02


# ============================================================
# MAIN
# ============================================================

def main():

    rospy.init_node("vr_left_arm_teleop")

    # --------------------------------------------------------
    # HEADSET
    # --------------------------------------------------------

    headset = WebRTCHeadset()
    headset.run_in_thread()

    # --------------------------------------------------------
    # ROBOT MODEL
    # --------------------------------------------------------

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    LEFT_ARM_NAMES = [
        "leftwaist",
        "leftshoulder",
        "leftelbow",
        "leftforearm_roll",
        "leftwrist_angle",
        "leftwrist_rotate",
    ]

    left_arm_indices = [
        robot.joints.actuated_names.index(name)
        for name in LEFT_ARM_NAMES
    ]

    # --------------------------------------------------------
    # INTERBOTIX
    # --------------------------------------------------------

    bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    print("interbotix joints:")
    print(bot.arm.group_info.joint_names)

    # --------------------------------------------------------
    # INITIAL TARGET
    # --------------------------------------------------------

    T_robot_target = np.eye(4)

    # MUCH SAFER STARTING POSE
    T_robot_target[:3,3] = np.array([
        0.25,
        -0.10,
        0.25,
    ])

    initial_rot = R.from_euler(
        "xyz",
        #[0, 90, 0],
        [-90, 0, -90],
        degrees=True,
    )

    quat_xyzw = initial_rot.as_quat()

    R_robot_target = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    # --------------------------------------------------------
    # IK STATE
    # --------------------------------------------------------

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    filtered_q = np.zeros(6)

    prev_cmd_q = None

    # --------------------------------------------------------
    # TELEOP STATE
    # --------------------------------------------------------

    teleop_active = False

    start_controller_pose = None
    start_robot_pose = None

    print("\nREADY")
    print("hold RIGHT BUTTON ONE to teleoperate\n")

    # ========================================================
    # LOOP
    # ========================================================

    while not rospy.is_shutdown():

        loop_start = time.time()

        # ----------------------------------------------------
        # RECEIVE HEADSET DATA
        # ----------------------------------------------------

        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        current_controller = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        button_pressed = (
            headset_data.r_button_one
        )

        # ----------------------------------------------------
        # TELEOP ENABLE
        # ----------------------------------------------------

        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            aligned_controller = np.eye(4)

            aligned_controller[:3,:3] = (
                align_rotation_to_z_axis(
                    current_controller[:3,:3]
                )
            )

            aligned_controller[:3,3] = (
                current_controller[:3,3]
            )

            start_controller_pose = (
                aligned_controller.copy()
            )

            start_robot_pose = (
                T_robot_target.copy()
            )

            print("\nTeleop ENABLED")

        # ----------------------------------------------------
        # TELEOP DISABLE
        # ----------------------------------------------------

        elif (
            not button_pressed
            and teleop_active
        ):

            teleop_active = False

            print("\nTeleop DISABLED")

        # ----------------------------------------------------
        # TELEOP
        # ----------------------------------------------------

        if teleop_active:

            # ------------------------------------------------
            # CONTROLLER DELTA
            # ------------------------------------------------

            delta = (
                current_controller[:3,3]
                - start_controller_pose[:3,3]
            )

            delta *= POSITION_SCALE

            # ------------------------------------------------
            # TARGET POSITION
            # ------------------------------------------------

            T_robot_target[:3,3] = (
                start_robot_pose[:3,3]
                + delta
            )

            # ------------------------------------------------
            # WORKSPACE CLAMP
            # ------------------------------------------------

            # forward/back

            T_robot_target[0,3] = np.clip(
                T_robot_target[0,3],
                0.15,
                0.40,
            )

            # left/right

            T_robot_target[1,3] = np.clip(
                T_robot_target[1,3],
                -0.25,
                0.25,
            )

            # up/down

            T_robot_target[2,3] = np.clip(
                T_robot_target[2,3],
                0.15,
                0.40,
            )

            # ------------------------------------------------
            # IK
            # ------------------------------------------------

            # IMPORTANT:
            # NO ORIENTATION CONSTRAINTS YET

            q_new = pks.solve_ik(
                robot=robot,
                target_link_name=LEFT_EE_LINK,
                target_position=T_robot_target[:3,3],
                target_wxyz=R_robot_target,
            )

            # ------------------------------------------------
            # VALID IK
            # ------------------------------------------------

            if q_new is not None:

                q = q_new

                left_arm_q = q[
                    left_arm_indices
                ]

                # --------------------------------------------
                # LOW PASS FILTER
                # --------------------------------------------

                filtered_q = (
                    ALPHA * left_arm_q
                    + (1 - ALPHA) * filtered_q
                )

                # --------------------------------------------
                # COMMAND DEADBAND
                # --------------------------------------------

                send_command = False

                if prev_cmd_q is None:
                    send_command = True

                else:

                    diff = np.linalg.norm(
                        filtered_q - prev_cmd_q
                    )

                    if diff > COMMAND_DEADBAND:
                        send_command = True

                # --------------------------------------------
                # SEND COMMAND
                # --------------------------------------------

                if send_command:

                    bot.arm.set_joint_positions(
                        filtered_q.tolist(),
                        moving_time=MOVING_TIME,
                        accel_time=ACCEL_TIME,
                        blocking=False,
                    )

                    prev_cmd_q = (
                        filtered_q.copy()
                    )

                    print(
                        "target:",
                        np.round(
                            T_robot_target[:3,3],
                            3,
                        )
                    )

                    print(
                        "cmd:",
                        np.round(
                            filtered_q,
                            3,
                        )
                    )

            else:

                print("IK FAILED")

        # ----------------------------------------------------
        # LOOP TIMING
        # ----------------------------------------------------

        elapsed = (
            time.time() - loop_start
        )

        time.sleep(
            max(
                0,
                CONTROL_DT - elapsed
            )
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
