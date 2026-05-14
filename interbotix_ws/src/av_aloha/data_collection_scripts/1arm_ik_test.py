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


# CONFIG

URDF_PATH = "/home/devi/giava/giava.urdf"

RIGHT_EE_LINK = "rightgripper_base"

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

    # DEBUG FRAME
    controller_frame = server.scene.add_frame(
        "/vr/right_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    ee_frame = server.scene.add_frame(
        "/ee",
        axes_length=0.15,
        axes_radius=0.01,
    )

    # ROBOT STATE
    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    # INITIAL ROBOT TARGET
    T_robot_target = np.eye(4)

    T_robot_target[:3,3] = np.array([
        -0.2,
        0.0,
        0.45,
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

    teleop_active = False

    start_controller_pose = None
    start_robot_pose = None

    while True:

        start_time = time.time()

        # RECEIVE HEADSET DATA
        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        # CURRENT CONTROLLER POSE
        current_controller_right = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        # VISUALIZE RAW CONTROLLER FRAME
        controller_frame.position = (
            headset_data.r_pos
        )

        controller_frame.wxyz = (
            xyzw_to_wxyz(
                headset_data.r_quat
            )
        )

        # BUTTON STATE
        button_pressed = (
            headset_data.r_button_one
        )

        # TELEOP START
        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            # CALIBRATE CONTROLLER FRAME
            aligned_controller = np.eye(4)

            aligned_controller[:3,:3] = (
                align_rotation_to_z_axis(
                    current_controller_right[:3,:3]
                )
            )

            aligned_controller[:3,3] = (
                current_controller_right[:3,3]
            )

            # SAVE REFERENCE FRAMES
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

        # TELEOP STOP
        elif (
            not button_pressed
            and teleop_active
        ):

            teleop_active = False

            print("Teleop DISABLED")

        # RUN TELEOP
        if teleop_active:

            # MAP CONTROLLER MOTION INTO
            # ROBOT TARGET FRAME


            if teleop_active:

                # controller motion since teleop start

                delta = (
                    current_controller_right[:3,3]
                    - start_controller_pose[:3,3]
                )

                # remap controller axes

                R_remap = np.array([
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                ])

                delta = R_remap @ delta

                # apply to robot target

                T_robot_target[:3,3] = (
                    start_robot_pose[:3,3]
                    + delta
                )

                R_delta = (
                    start_controller_rot.T
                    @ current_controller_right[:3,:3]
                )

                controller_rpy = (
                    R.from_matrix(R_delta)
                    .as_euler("xyz", degrees=False)
                )

                rotvec = (
                    R.from_matrix(R_delta)
                    .as_rotvec()
                )

                print("controller semantic rpy")
                print(controller_rpy)

                pitch = rotvec[1]
                yaw   = rotvec[0]
                roll  = rotvec[2]

                # tried 012, 210, 120 (pitch makes sense but is directionally reversed!), 102 (this makes sense but yes pitch still dir rev)

                rotvec_ee = np.array([
                    -pitch,
                    yaw,
                    roll,
                ])

                R_local_delta = (
                    R.from_rotvec(rotvec_ee)
                    .as_matrix()
                )

                R_target = (
                    start_robot_rot
                    @ R_local_delta
                )
                
               # R_target = R_target @ R_offset

                # matrix -> quaternion
                # scipy gives xyzw----

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

                R_ee = R.from_quat([
                    ee_wxyz[1],
                    ee_wxyz[2],
                    ee_wxyz[3],
                    ee_wxyz[0],
                ]).as_matrix()

                R_error = (
                    R_target
                    @ R_ee.T
                )

                error_euler = (
                    R.from_matrix(R_error)
                    .as_euler("xyz", degrees=True)
                )

                """ controller_x = current_controller_right[:3,0]
                controller_y = current_controller_right[:3,1]
                controller_z = current_controller_right[:3,2]

                ee_x = R_ee[:,0]
                ee_y = R_ee[:,1]
                ee_z = R_ee[:,2]

                print("\nCONTROLLER AXES")
                print("x:", controller_x)
                print("y:", controller_y)
                print("z:", controller_z)

                print("\nEE AXES")
                print("x:", ee_x)
                print("y:", ee_y)
                print("z:", ee_z) """

        # SOLVE IK

        q_new = pks.solve_ik(
            robot=robot,
            target_link_name=RIGHT_EE_LINK,

            # POSITION TARGET

            target_position=(
                T_robot_target[:3,3]
            ),

            # TARGET ORIENTATION

            target_wxyz=R_robot_target,
        )

        # UPDATE ROBOT STATE

        if q_new is not None:
            q = q_new

            fk = robot.forward_kinematics(q)

            ee_index = robot.links.names.index(
                RIGHT_EE_LINK
            )

            ee_pose = np.array(
                fk[ee_index]
            )

            ee_wxyz = ee_pose[:4]
            ee_position = ee_pose[4:]


            ee_frame.position = ee_position

            ee_frame.wxyz = ee_wxyz

        # UPDATE VISUALIZATION

        joint_dict = {
            name: value
            for name, value in zip(
                robot.joints.actuated_names,
                q,
            )
        }

        

        if headset_data.r_hand_trigger > 0.001:
            print(f"right trigger is pressed! {headset_data.r_hand_trigger}")

            gripper_value = 0.0

        else:

            gripper_value = 0.041

        joint_dict["rightright_finger"] = gripper_value

        for k, v in vars(headset_data).items():
            if isinstance(v, (int, float)):
                print(k, v)

        urdf_vis.update_cfg(
            joint_dict
        )

        time.sleep(0.01)

if __name__ == "__main__":
    main()