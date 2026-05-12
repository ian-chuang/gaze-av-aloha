import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

from viser.extras import ViserUrdf
from yourdfpy import URDF

import pyroki as pk
import sys

sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks

from webrtc_headset import WebRTCHeadset

from transform_utils import (
    pose2mat,
    transform_coordinates,
    align_rotation_to_z_axis,
    xyzw_to_wxyz,
)

# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

RIGHT_EE_LINK = "rightgripper_base"

# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADSET
    # ========================================================

    headset = WebRTCHeadset()
    headset.run_in_thread()

    # ========================================================
    # ROBOT
    # ========================================================

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    # ========================================================
    # VISER
    # ========================================================

    server = viser.ViserServer()

    server.scene.add_grid(
        "/ground",
        width=2,
        height=2,
    )

    urdf_vis = ViserUrdf(
        server,
        urdf,
        root_node_name="/base",
    )

    # ========================================================
    # DEBUG FRAME
    # ========================================================

    controller_frame = server.scene.add_frame(
        "/vr/right_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    # ========================================================
    # ROBOT STATE
    # ========================================================

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    # ========================================================
    # INITIAL ROBOT TARGET
    # ========================================================

    T_robot_target = np.eye(4)

    T_robot_target[:3,3] = np.array([
        -0.2,
        0.0,
        0.45,
    ])

    initial_rot = R.from_euler(
        "xyz",
        #[0, 90, 0],
        [90, 0, 90],
        degrees=True,
    )

    quat_xyzw = initial_rot.as_quat()

    R_robot_target = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    # # ========================================================
    # # TELEOP FRAME CHANGE
    # # ========================================================

    # T_teleop_frame = np.eye(4)

    # T_teleop_frame[:3,:3] = np.array([
    #     [ 0,  1,  0],
    #     [-1,  0,  0],
    #     [ 0,  0,  1],
    # ])

    # T_teleop_frame[:3,3] = (
    #     T_robot_target[:3,3]
    # )

    # ========================================================
    # TELEOP STATE
    # ========================================================

    teleop_active = False

    start_controller_pose = None
    start_robot_pose = None

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        start_time = time.time()

        # ----------------------------------------------------
        # RECEIVE HEADSET DATA
        # ----------------------------------------------------

        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        # ----------------------------------------------------
        # CURRENT CONTROLLER POSE
        # ----------------------------------------------------

        current_controller_right = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        # ----------------------------------------------------
        # VISUALIZE RAW CONTROLLER FRAME
        # ----------------------------------------------------

        controller_frame.position = (
            headset_data.r_pos
        )

        controller_frame.wxyz = (
            xyzw_to_wxyz(
                headset_data.r_quat
            )
        )

        # ----------------------------------------------------
        # BUTTON STATE
        # ----------------------------------------------------

        button_pressed = (
            headset_data.r_button_one
        )

        # ====================================================
        # TELEOP START
        # ====================================================

        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            # ------------------------------------------------
            # CALIBRATE CONTROLLER FRAME
            # ------------------------------------------------

            aligned_controller = np.eye(4)

            aligned_controller[:3,:3] = (
                align_rotation_to_z_axis(
                    current_controller_right[:3,:3]
                )
            )

            aligned_controller[:3,3] = (
                current_controller_right[:3,3]
            )

            # ------------------------------------------------
            # SAVE REFERENCE FRAMES
            # ------------------------------------------------

            start_controller_pose = (
                aligned_controller.copy()
            )

            start_robot_pose = (
                T_robot_target.copy()
            )

            start_controller_rot = (
                current_controller_right[:3,:3].copy()
            )

            start_robot_rot = (
                R.from_quat([
                    R_robot_target[1],
                    R_robot_target[2],
                    R_robot_target[3],
                    R_robot_target[0],
                ]).as_matrix()
            )

            print("Teleop ENABLED")

        # ====================================================
        # TELEOP STOP
        # ====================================================

        elif (
            not button_pressed
            and teleop_active
        ):

            teleop_active = False

            print("Teleop DISABLED")

        # ====================================================
        # RUN TELEOP
        # ====================================================

        if teleop_active:

            # ------------------------------------------------
            # MAP CONTROLLER MOTION INTO
            # ROBOT TARGET FRAME
            # ------------------------------------------------


            if teleop_active:

                # ----------------------------------------
                # controller motion since teleop start
                # ----------------------------------------

                delta = (
                    current_controller_right[:3,3]
                    - start_controller_pose[:3,3]
                )

                # ----------------------------------------
                # remap controller axes
                # ----------------------------------------

                R_remap = np.array([
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                ])

                delta = R_remap @ delta

                # ----------------------------------------
                # apply to robot target
                # ----------------------------------------

                T_robot_target[:3,3] = (
                    start_robot_pose[:3,3]
                    + delta
                )

                # --------------------------------------------
                # controller orientation
                # --------------------------------------------

                #R_target = current_controller_right[:3,:3]

                # --------------------------------------------
                # rotate gripper so it faces forward
                # --------------------------------------------

                # R_offset = np.array([
                #     [0, 0, 1],
                #     [1, 0, 0],
                #     [0, 1, 0],
                # ])

                # yaw = np.deg2rad(-45)
                # roll = np.deg2rad(-90)

                # R_yaw = np.array([
                #     [ np.cos(yaw), -np.sin(yaw), 0],
                #     [ np.sin(yaw),  np.cos(yaw), 0],
                #     [ 0,             0,          1],
                # ])

                # R_roll = np.array([
                #     [1, 0,              0             ],
                #     [0, np.cos(roll),  -np.sin(roll)],
                #     [0, np.sin(roll),   np.cos(roll)],
                # ])

                # R_offset = R_yaw @ R_roll

                # ----------------------------------------
                # relative controller rotation
                # ----------------------------------------

                

                R_delta = (
                    start_controller_rot.T
                    @ current_controller_right[:3,:3]
                )

                # R_axis_remap = np.array([
                #     [0, 1, 0],
                #     [0, 0, -1],
                #     [1, 0, 0],
                # ])

                R_axis_remap = np.array([
                    [0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0],
                ])

                R_delta_remapped = (
                    R_axis_remap
                    @ R_delta
                    @ R_axis_remap.T
                )

                # ----------------------------------------
                # apply calibration offset
                # ----------------------------------------

                R_target = (
                    start_robot_rot
                    @ R_delta_remapped
                    #@ R_offset
                )
                
               # R_target = R_target @ R_offset

                # --------------------------------------------
                # matrix -> quaternion
                # scipy gives xyzw
                # --------------------------------------------

                quat_xyzw = (
                    R.from_matrix(R_target)
                    .as_quat()
                )

                R_robot_target = np.array([
                    quat_xyzw[3],
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                ])

            # T_robot_target = (
            #     transform_coordinates(
            #         current_controller_right,
            #         start_controller_pose,
            #         start_robot_pose,
            #     )
            # )

        # ====================================================
        # SOLVE IK
        # ====================================================

        q_new = pks.solve_ik(
            robot=robot,
            target_link_name=RIGHT_EE_LINK,

            # --------------------------------------------
            # POSITION TARGET
            # --------------------------------------------

            target_position=(
                T_robot_target[:3,3]
            ),

            # --------------------------------------------
            # TARGET ORIENTATION
            # --------------------------------------------

            target_wxyz=R_robot_target,
        )

        # ----------------------------------------------------
        # UPDATE ROBOT STATE
        # ----------------------------------------------------

        if q_new is not None:
            q = q_new

        # ====================================================
        # UPDATE VISUALIZATION
        # ====================================================

        joint_dict = {
            name: value
            for name, value in zip(
                robot.joints.actuated_names,
                q,
            )
        }

        urdf_vis.update_cfg(
            joint_dict
        )

        # ====================================================
        # DEBUG
        # ====================================================

        elapsed_ms = (
            time.time()
            - start_time
        ) * 1000.0

        print(
            f"teleop={teleop_active} "
            f"target={T_robot_target[:3,3]} "
            f"dt={elapsed_ms:.1f}ms"
        )

        time.sleep(0.01)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()