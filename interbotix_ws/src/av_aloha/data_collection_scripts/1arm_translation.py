import rospy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from interbotix_xs_modules.arm import (
    InterbotixManipulatorXS
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


# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "leftgripper_base"

CONTROL_DT = 0.05

MOVING_TIME = 0.07
ACCEL_TIME = 0.02

POSITION_SCALE = 1.0

ALPHA = 0.2

# ============================================================
# MAIN
# ============================================================

def main():

    rospy.init_node(
        "vr_cartesian_teleop"
    )

    # --------------------------------------------------------
    # HEADSET
    # --------------------------------------------------------

    headset = WebRTCHeadset()

    headset.run_in_thread()

    # --------------------------------------------------------
    # ROBOT MODEL
    # --------------------------------------------------------

    urdf = URDF.load(
        URDF_PATH
    )

    robot = pk.Robot.from_urdf(
        urdf
    )

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

    ee_index = robot.links.names.index(
        LEFT_EE_LINK
    )

    # --------------------------------------------------------
    # INTERBOTIX
    # --------------------------------------------------------

    bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_left",
        moving_time=MOVING_TIME,
        accel_time=ACCEL_TIME,
        init_node=False,
    )

    rospy.sleep(1.0)

    # --------------------------------------------------------
    # CURRENT ROBOT STATE
    # --------------------------------------------------------

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    current_q = np.array(
        bot.arm.get_joint_commands()
    )

    q[left_arm_indices] = current_q

    # --------------------------------------------------------
    # FK
    # --------------------------------------------------------

    fk = robot.forward_kinematics(q)

    T = jaxlie.SE3(
        fk[ee_index]
    ).as_matrix()

    start_robot_position = np.array(
        T[:3,3].copy()
    )

    filtered_target_position = (
        start_robot_position.copy()
    )

    print("\nInitial EE position:")
    print(start_robot_position)

    # --------------------------------------------------------
    # FIXED ORIENTATION
    # --------------------------------------------------------

    fixed_rot = R.from_euler(
        "xyz",
        [-90, 0, -90],
        degrees=True,
    )

    quat_xyzw = fixed_rot.as_quat()

    target_wxyz = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    # --------------------------------------------------------
    # VELOCITY LIMITS
    # --------------------------------------------------------

    full_joint_velocity_limits = np.zeros(
        robot.joints.num_actuated_joints
    )

    full_joint_velocity_limits[
        left_arm_indices
    ] = np.array([
        2.0,
        2.0,
        2.0,
        4.0,
        4.0,
        4.0,
    ])

    # ========================================================
    # CONTROLLER -> ROBOT FRAME MAP
    # ========================================================

    R_remap = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ])

    # ========================================================
    # TELEOP STATE
    # ========================================================

    teleop_active = False

    start_controller_position = None

    # ========================================================
    # LOOP
    # ========================================================

    print("\nREADY")
    print(
        "Hold LEFT BUTTON ONE to teleoperate\n"
    )

    while not rospy.is_shutdown():

        headset_data = (
            headset.receive_data()
        )

        if headset_data is None:

            time.sleep(0.01)

            continue

        current_controller = pose2mat(
            headset_data.l_pos,
            headset_data.l_quat,
        )

        button_pressed = (
            headset_data.l_button_one
        )

        # ----------------------------------------------------
        # ENABLE TELEOP
        # ----------------------------------------------------

        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            start_controller_position = (
                current_controller[:3,3].copy()
            )

            start_robot_position = np.array(
                T[:3,3]
            )

            print("\nTeleop ENABLED")

        # ----------------------------------------------------
        # DISABLE TELEOP
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

            delta_controller = (

                current_controller[:3,3]

                - start_controller_position
            )

            # ------------------------------------------------
            # MAP INTO ROBOT FRAME
            # ------------------------------------------------

            delta_robot = (
                R_remap
                @ delta_controller
            )

            # ------------------------------------------------
            # TARGET POSITION
            # ------------------------------------------------

            # target_position = (

            #     start_robot_position

            #     + POSITION_SCALE
            #     * delta_robot
            # )

            raw_target_position = (
                start_robot_position
                + POSITION_SCALE * delta_robot
            )

            filtered_target_position = (

                ALPHA * raw_target_position

                + (1 - ALPHA)
                * filtered_target_position
            )

            target_position = (
                filtered_target_position
            )

            # ------------------------------------------------
            # WORKSPACE CLAMP
            # ------------------------------------------------

            target_position[0] = np.clip(
                target_position[0],
                0.15,
                0.45,
            )

            target_position[1] = np.clip(
                target_position[1],
                -0.35,
                0.35,
            )

            target_position[2] = np.clip(
                target_position[2],
                0.10,
                0.50,
            )

            # ------------------------------------------------
            # IK
            # ------------------------------------------------

            q_new = pks.solve_trajectory_ik(

                robot=robot,

                target_link_name=LEFT_EE_LINK,

                target_position=target_position,

                target_wxyz=target_wxyz,

                prev_q=q,

                dt=CONTROL_DT,

                joint_velocity_limits=(
                    full_joint_velocity_limits
                ),

                left_arm_indices=(
                    left_arm_indices
                ),
            )

            # ------------------------------------------------
            # VALID IK
            # ------------------------------------------------

            if q_new is not None:

                q = q_new

                left_q = q[
                    left_arm_indices
                ]

                bot.arm.set_joint_positions(

                    left_q.tolist(),

                    moving_time=MOVING_TIME,

                    accel_time=ACCEL_TIME,

                    blocking=False,
                )

                # update FK state

                fk = robot.forward_kinematics(q)

                T = jaxlie.SE3(
                    fk[ee_index]
                ).as_matrix()

                print(
                    "target:",
                    np.round(
                        target_position,
                        3
                    )
                )

            else:

                print("IK FAILED")

        time.sleep(CONTROL_DT)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()