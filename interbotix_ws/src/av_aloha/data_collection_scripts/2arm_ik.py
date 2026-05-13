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
    align_rotation_to_z_axis,
    xyzw_to_wxyz,
)

# CONFIG

URDF_PATH = "/home/devi/giava/giava.urdf"

RIGHT_EE_LINK = "rightgripper_base"
LEFT_EE_LINK  = "leftgripper_base"

# HELPERS

def wxyz_from_matrix(R_mat):

    quat_xyzw = (
        R.from_matrix(R_mat)
        .as_quat()
    )

    return np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])


def matrix_from_wxyz(wxyz):

    return R.from_quat([
        wxyz[1],
        wxyz[2],
        wxyz[3],
        wxyz[0],
    ]).as_matrix()

# MAIN

def main():

    # HEADSET

    headset = WebRTCHeadset()
    headset.run_in_thread()

    # ROBOT

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    # VISER

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

    # DEBUG FRAMES

    right_controller_frame = server.scene.add_frame(
        "/vr/right_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    left_controller_frame = server.scene.add_frame(
        "/vr/left_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    right_ee_frame = server.scene.add_frame(
        "/right_ee",
        axes_length=0.15,
        axes_radius=0.01,
    )

    left_ee_frame = server.scene.add_frame(
        "/left_ee",
        axes_length=0.15,
        axes_radius=0.01,
    )

    # ROBOT STATE

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    # RIGHT ARM INITIAL TARGET

    T_right_target = np.eye(4)

    T_right_target[:3,3] = np.array([
        -0.2,
        -0.25,
        0.45,
    ])

    right_initial_rot = R.from_euler(
        "xyz",
        [-90, 0, -90],
        degrees=True,
    )

    R_right_target = wxyz_from_matrix(
        right_initial_rot.as_matrix()
    )

    # LEFT ARM INITIAL TARGET

    T_left_target = np.eye(4)

    T_left_target[:3,3] = np.array([
        0.1,
        0,
        0.45,
    ])

    left_initial_rot = R.from_euler(
        "xyz",
        [-90, 0, 90],
        degrees=True,
    )

    R_left_target = wxyz_from_matrix(
        left_initial_rot.as_matrix()
    )

    # TELEOP STATE

    teleop_active = False

    # RIGHT

    right_start_controller_pose = None
    right_start_robot_pose = None

    right_start_controller_rot = None
    right_start_robot_rot = None

    # LEFT

    left_start_controller_pose = None
    left_start_robot_pose = None

    left_start_controller_rot = None
    left_start_robot_rot = None

    # TRANSLATION REMAP

    R_remap = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])

    # FK INDICES

    right_index = robot.links.names.index(
        RIGHT_EE_LINK
    )

    left_index = robot.links.names.index(
        LEFT_EE_LINK
    )

    # MAIN LOOP

    while True:

        start_time = time.time()

        # RECEIVE DATA

        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        # CONTROLLER POSES

        current_controller_right = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        current_controller_left = pose2mat(
            headset_data.l_pos,
            headset_data.l_quat,
        )

        # VISUALIZE CONTROLLERS

        right_controller_frame.position = (
            headset_data.r_pos
        )

        right_controller_frame.wxyz = (
            xyzw_to_wxyz(
                headset_data.r_quat
            )
        )

        left_controller_frame.position = (
            headset_data.l_pos
        )

        left_controller_frame.wxyz = (
            xyzw_to_wxyz(
                headset_data.l_quat
            )
        )

        # BUTTONS

        teleop_button = (
            headset_data.r_button_one
        )

        # TELEOP START

        if teleop_button and not teleop_active:

            teleop_active = True

            # RIGHT ARM CALIBRATION

            aligned_controller = np.eye(4)

            aligned_controller[:3,:3] = (
                align_rotation_to_z_axis(
                    current_controller_right[:3,:3]
                )
            )

            aligned_controller[:3,3] = (
                current_controller_right[:3,3]
            )

            right_start_controller_pose = (
                aligned_controller.copy()
            )

            right_start_robot_pose = (
                T_right_target.copy()
            )

            right_start_controller_rot = (
                current_controller_right[:3,:3].copy()
            )

            right_start_robot_rot = (
                matrix_from_wxyz(
                    R_right_target
                )
            )

            # LEFT ARM CALIBRATION

            aligned_controller = np.eye(4)

            aligned_controller[:3,:3] = (
                align_rotation_to_z_axis(
                    current_controller_left[:3,:3]
                )
            )

            aligned_controller[:3,3] = (
                current_controller_left[:3,3]
            )

            left_start_controller_pose = (
                aligned_controller.copy()
            )

            left_start_robot_pose = (
                T_left_target.copy()
            )

            left_start_controller_rot = (
                current_controller_left[:3,:3].copy()
            )

            left_start_robot_rot = (
                matrix_from_wxyz(
                    R_left_target
                )
            )

            print("BIMANUAL TELEOP ENABLED")

        elif (not teleop_button) and teleop_active:

            teleop_active = False

            print("BIMANUAL TELEOP DISABLED")

        # TELEOP

        if teleop_active:

            # RIGHT TELEOP

            # TRANSLATION

            delta_right = (
                current_controller_right[:3,3]
                - right_start_controller_pose[:3,3]
            )

            delta_right = R_remap @ delta_right

            T_right_target[:3,3] = (
                right_start_robot_pose[:3,3]
                + delta_right
            )

            # ROTATION

            R_delta_right = (
                right_start_controller_rot.T
                @ current_controller_right[:3,:3]
            )

            rotvec_right = (
                R.from_matrix(R_delta_right)
                .as_rotvec()
            )

            pitch_right = rotvec_right[1]
            yaw_right   = rotvec_right[0]
            roll_right  = rotvec_right[2]

            rotvec_ee_right = np.array([
                -pitch_right,
                yaw_right,
                roll_right,
            ])

            R_local_delta_right = (
                R.from_rotvec(rotvec_ee_right)
                .as_matrix()
            )

            R_target_right = (
                right_start_robot_rot
                @ R_local_delta_right
            )

            R_right_target = (
                wxyz_from_matrix(
                    R_target_right
                )
            )

            # LEFT TELEOP

            # TRANSLATION

            delta_left = (
                current_controller_left[:3,3]
                - left_start_controller_pose[:3,3]
            )

            delta_left = R_remap @ delta_left

            T_left_target[:3,3] = (
                left_start_robot_pose[:3,3]
                + delta_left
            )

            # ROTATION

            R_delta_left = (
                left_start_controller_rot.T
                @ current_controller_left[:3,:3]
            )

            rotvec_left = (
                R.from_matrix(R_delta_left)
                .as_rotvec()
            )

            print("\nLEFT CONTROLLER ROTVEC")
            print(rotvec_left)

            print("x component:", rotvec_left[0])
            print("y component:", rotvec_left[1])
            print("z component:", rotvec_left[2])

            pitch_left = rotvec_left[1]
            yaw_left   = rotvec_left[0]
            roll_left  = rotvec_left[2]

            rotvec_ee_left = np.array([
                -pitch_left,
                yaw_left,
                roll_left,
            ])

            R_local_delta_left = (
                R.from_rotvec(rotvec_ee_left)
                .as_matrix()
            )

            R_target_left = (
                left_start_robot_rot
                @ R_local_delta_left
            )

            R_left_target = (
                wxyz_from_matrix(
                    R_target_left
                )
            )

            # IK

            solution = pks.solve_ik_with_multiple_targets(
                robot=robot,

                target_link_names=[
                    RIGHT_EE_LINK,
                    LEFT_EE_LINK,
                ],

                target_positions=np.array([
                    T_right_target[:3,3],
                    T_left_target[:3,3],
                ]),

                target_wxyzs=np.array([
                    R_right_target,
                    R_left_target,
                ]),
            )

            if solution is not None:
                q = solution

        # FK

        fk = robot.forward_kinematics(q)

        # RIGHT EE

        right_pose = np.array(
            fk[right_index]
        )

        right_ee_frame.wxyz = (
            right_pose[:4]
        )

        right_ee_frame.position = (
            right_pose[4:]
        )

        # LEFT EE

        left_pose = np.array(
            fk[left_index]
        )

        left_ee_frame.wxyz = (
            left_pose[:4]
        )

        left_ee_frame.position = (
            left_pose[4:]
        )

        # JOINT DICT

        joint_dict = {
            name: value
            for name, value in zip(
                robot.joints.actuated_names,
                q,
            )
        }

        # RIGHT GRIPPER

        if headset_data.r_hand_trigger > 0.5:

            right_gripper = 0.0

        else:

            right_gripper = 0.041

        joint_dict["rightright_finger"] = (
            right_gripper
        )

        joint_dict["rightleft_finger"] = (
            right_gripper
        )

        # LEFT GRIPPER

        if headset_data.l_hand_trigger > 0.5:

            left_gripper = 0.0

        else:

            left_gripper = 0.041

        joint_dict["leftright_finger"] = (
            left_gripper
        )

        joint_dict["leftleft_finger"] = (
            left_gripper
        )

        # UPDATE VISER

        urdf_vis.update_cfg(
            joint_dict
        )

        dt = (
            time.time() - start_time
        ) * 1000

        print(
            f"teleop={teleop_active} dt={dt:.1f}ms"
        )

        time.sleep(0.01)


if __name__ == "__main__":
    main()
