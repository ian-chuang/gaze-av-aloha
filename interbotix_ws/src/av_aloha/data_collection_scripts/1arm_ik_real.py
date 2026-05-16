


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

    """     print("\nMoving to teleop start pose...")

    startup_q = [
        0.0,    # waist
    -0.45,   # shoulder
        1.0,    # elbow
        0.0,    # forearm_roll
    -0.55,   # wrist_angle
        0.0,    # wrist_rotate
    ]

    bot.arm.set_joint_positions(
        startup_q,
        moving_time=3.0,
        accel_time=0.8,
        blocking=True,
    )

    time.sleep(1.0)

    print("Reached teleop start pose") """

    # --------------------------------------------------------
    # INITIAL TARGET
    # --------------------------------------------------------

    bot.arm.set_joint_positions(
        [0, 0.5, 0, 0, 0, 0],
        blocking=True,
    )

    T_robot_target = np.eye(4)

    # MUCH SAFER STARTING POSE
    T_robot_target[:3,3] = np.array([
        0.25,
        -0.10,
        0.25,
    ])

    initial_rot = R.from_euler(
        "xyz",
        [-90, 0, 90],
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
    start_controller_rot = None
    start_robot_rot = None

    prev_delta_robot = np.zeros(3)

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
            headset_data.l_pos,
            headset_data.l_quat,
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

            start_controller_rot = (
                current_controller[:3,:3].copy()
            )

            start_robot_rot = (
                R.from_quat([
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                    quat_xyzw[3],
                ]).as_matrix()
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
            # REMAP CONTROLLER FRAME -> ROBOT FRAME
            # ------------------------------------------------

            delta_robot = np.array([
                -delta[0],
                -delta[1],
                delta[2],
            ])

            # ------------------------------------------------
            # INCREMENTAL MOTION
            # ------------------------------------------------

            delta_step = (
                delta_robot - prev_delta_robot
            )

            prev_delta_robot = delta_robot.copy()

            # ------------------------------------------------
            # CARTESIAN VELOCITY LIMIT
            # ------------------------------------------------

            MAX_STEP = 0.01

            step_norm = np.linalg.norm(delta_step)

            if step_norm > MAX_STEP:

                delta_step = (
                    delta_step
                    / step_norm
                    * MAX_STEP
                )

            # ------------------------------------------------
            # INTEGRATE TARGET
            # ------------------------------------------------

            T_robot_target[:3,3] += delta_step

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

            # ====================================================
            # RELATIVE ROTATION CONTROL
            # ====================================================

            R_delta = (
                start_controller_rot.T
                @ current_controller[:3,:3]
            )

            rotvec = (
                R.from_matrix(R_delta)
                .as_rotvec()
            )

            # controller axis remap

            pitch = rotvec[1]
            yaw   = rotvec[0]
            roll  = rotvec[2]

            rotvec_ee = np.array([
                -pitch,
                yaw,
                roll,
            ])

            # ROTATION LIMIT

            MAX_ROT_STEP = 0.15

            rot_norm = np.linalg.norm(rotvec_ee)

            if rot_norm > MAX_ROT_STEP:

                rotvec_ee = (
                    rotvec_ee
                    / rot_norm
                    * MAX_ROT_STEP
                )

            # smoothing

            ROT_ALPHA = 0.15

            rotvec_ee *= ROT_ALPHA

            R_local_delta = (
                R.from_rotvec(rotvec_ee)
                .as_matrix()
            )

            R_target = (
                start_robot_rot
                @ R_local_delta
            )

            quat_xyzw_target = (
                R.from_matrix(R_target)
                .as_quat()
            )

            target_wxyz = np.array([
                quat_xyzw_target[3],
                quat_xyzw_target[0],
                quat_xyzw_target[1],
                quat_xyzw_target[2],
            ])

            # ------------------------------------------------
            # IK
            # ------------------------------------------------

            # IMPORTANT:
            # NO ORIENTATION CONSTRAINTS YET

            q_new = pks.solve_ik(
                robot=robot,
                target_link_name=LEFT_EE_LINK,
                target_position=T_robot_target[:3,3],
                target_wxyz=target_wxyz,
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
# 
# 
# import time
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import rospy
# import pyroki as pk
# import sys
# sys.path.append("/home/devi/giava/pyroki/examples")
# import pyroki_snippets as pks
# from yourdfpy import URDF
# from webrtc_headset import WebRTCHeadset

# from transform_utils import (
#     pose2mat,
#     transform_coordinates,
#     align_rotation_to_z_axis,
#     xyzw_to_wxyz,
# )

# from interbotix_xs_modules.arm import InterbotixManipulatorXS

# # from sensor_msgs.msg import JointState
# # from interbotix_xs_msgs.msg import JointGroupCommand

# # CONFIG

# URDF_PATH = "/home/devi/giava/giava.urdf"

# LEFT_EE_LINK  = "leftgripper_base"

# CONTROL_DT = 0.02

# def wxyz_from_matrix(R_mat):

#     quat_xyzw = (
#         R.from_matrix(R_mat)
#         .as_quat()
#     )

#     return np.array([
#         quat_xyzw[3],
#         quat_xyzw[0],
#         quat_xyzw[1],
#         quat_xyzw[2],
#     ])


# def matrix_from_wxyz(wxyz):

#     return R.from_quat([
#         wxyz[1],
#         wxyz[2],
#         wxyz[3],
#         wxyz[0],
#     ]).as_matrix()

# def main():

#     # HEADSET
#     headset = WebRTCHeadset()
#     headset.run_in_thread()

#     # ROBOT
#     urdf = URDF.load(URDF_PATH)
#     robot = pk.Robot.from_urdf(urdf)

#     # INDICES

#     LEFT_ARM_NAMES = [
#         "leftwaist",
#         "leftshoulder",
#         "leftelbow",
#         "leftforearm_roll",
#         "leftwrist_angle",
#         "leftwrist_rotate",
#     ]

#     left_arm_indices = [
#         robot.joints.actuated_names.index(name)
#         for name in LEFT_ARM_NAMES
#     ]

#     # ROBOT STATE
#     q = np.zeros(
#         robot.joints.num_actuated_joints
#     )

#     # INITIAL ROBOT TARGET
#     T_robot_target = np.eye(4)

#     T_robot_target[:3,3] = np.array([
#         0.1,
#         0,
#         0.45,
#     ])


#     initial_rot = R.from_euler(
#         "xyz",
#         #[0, 90, 0],
#         [-90, 0, 90],
#         degrees=True,
#     )

#     quat_xyzw = initial_rot.as_quat()

#     R_robot_target = np.array([
#         quat_xyzw[3],
#         quat_xyzw[0],
#         quat_xyzw[1],
#         quat_xyzw[2],
#     ])

#     teleop_active = False

#     start_controller_pose = None
#     start_robot_pose = None

#     rospy.init_node("vr_left_arm_teleop")

#     bot = InterbotixManipulatorXS(
#         robot_model="wx250s",
#         group_name="arm",
#         gripper_name="gripper",
#         robot_name="puppet_left",
#         moving_time=0.10,#0.05,
#         accel_time=0.03,#0.01,
#         init_node=False,
#     )

#     t = 0.02

#     # TELEOP STATE

#     # LEFT

#     left_start_controller_pose = None
#     left_start_robot_pose = None

#     left_start_controller_rot = None
#     left_start_robot_rot = None

#     # TRANSLATION REMAP

#     # R_remap = np.array([
#     #     [0, 1, 0],
#     #     [-1, 0, 0],
#     #     [0, 0, 1],
#     # ])

#     current_joint_state = None

#     def joint_state_callback(msg):
#         global current_joint_state
#         current_joint_state = msg

#     while not rospy.is_shutdown():

#         # start_time = time.time()

#         # RECEIVE HEADSET DATA
#         headset_data = headset.receive_data()

#         if headset_data is None:
#             continue

#         # CURRENT CONTROLLER POSE
#         current_controller_right = pose2mat( headset_data.r_pos, headset_data.r_quat, )

#         # BUTTON STATE
#         button_pressed = ( headset_data.r_button_one )

#         # TELEOP START
#         if ( button_pressed and not teleop_active ):

#             teleop_active = True

#             # CALIBRATE CONTROLLER FRAME
#             aligned_controller = np.eye(4)

#             aligned_controller[:3,:3] = (
#                 align_rotation_to_z_axis(
#                     current_controller_right[:3,:3]
#                 )
#             )

#             aligned_controller[:3,3] = (
#                 current_controller_right[:3,3]
#             )

#             # SAVE REFERENCE FRAMES
#             start_controller_pose = (
#                 aligned_controller.copy()
#             )

#             start_robot_pose = (
#                 T_robot_target.copy()
#             )

#             start_controller_rot = (
#                 current_controller_right[:3,:3].copy()
#             )

#             start_robot_rot = (
#                 R.from_quat([
#                     R_robot_target[1],
#                     R_robot_target[2],
#                     R_robot_target[3],
#                     R_robot_target[0],
#                 ]).as_matrix()
#             )

#             print("Teleop ENABLED")

#         # TELEOP STOP
#         elif ( not button_pressed and teleop_active ):

#             teleop_active = False

#             print("Teleop DISABLED")

#         # RUN TELEOP

#         loop_start = time.time()

#         if teleop_active:

#             # MAP CONTROLLER MOTION INTO ROBOT TARGET FRAME
#             # controller motion since teleop start

#             delta = (
#                 current_controller_right[:3,3]
#                 - start_controller_pose[:3,3]
#             )

#             scale = 1.5
#             delta *= scale

#             T_robot_target[:3,3] = (
#                 start_robot_pose[:3,3]
#                 + delta
#             )

        
#             # SOLVE IK

#             """ q = np.zeros(23)

#             q[0] = np.sin(2*t)

#             bot.arm.set_joint_positions(
#                 q[8:14].tolist(),
#                 blocking=False,
#             )

#             t += 0.03 """

#             q_new = pks.solve_ik(
#                 robot=robot,
#                 target_link_name=LEFT_EE_LINK,

#                 target_position=T_robot_target[:3,3],

#                 # TEMPORARILY REMOVE ORIENTATION
#                 target_wxyz=R_robot_target,
#             )

#             print(robot.joints.actuated_names)

#             if q_new is not None:

#                 q_prev = q

#                 q = q_new

#                 left_arm_q = q[left_arm_indices]

#                 print("left arm q:", left_arm_q)

#                 # bot.arm.set_joint_positions(
#                 #     left_arm_q.tolist(),
#                 #     blocking=False,
#                 # )

#                 bot.arm.set_joint_positions(
#                     left_arm_q.tolist(),
#                     moving_time=0.15,
#                     accel_time=0.05,
#                     blocking=False,
#                 )

#                 print(np.linalg.norm(left_arm_q - q_prev[8:14]))

#             else:

#                 print("IK FAILED")

#             print("delta:", delta)
#             print("target:", T_robot_target[:3,3])

#             elapsed = time.time() - loop_start
#             time.sleep(max(0, CONTROL_DT - elapsed))

#         # POSITION TARGET

#         # TARGET ORIENTATION

#         # # GRIPPER CONTROL

#         for k, v in vars(headset_data).items():
#             if isinstance(v, (int, float)):
#                 print(k, v)

#         time.sleep(0.01)

# if __name__ == "__main__":
#     main()