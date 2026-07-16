"""
VR Teleoperation + Correct Tri-Arm Visualization

Uses:
- Your KNOWN GOOD Viser URDF visualization
- Headset/controller tracking
- Relative-pose clutch teleop
- Multi-target IK
- Stereo passthrough

Robot only moves while holding RIGHT controller button.
"""

import time
import numpy as np
import depthai as dai
import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf
from yourdfpy import URDF

import pyroki as pk
import sys

sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks

from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback

from transform_utils import (
    pose2mat,
    xyzw_to_wxyz,
)


# ============================================================
# CONFIG
# ============================================================

FPS = 25
V_MAX = 2.0

URDF_PATH = "/home/devi/giava/giava.urdf"

# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADSET
    # ========================================================

    headset = WebRTCHeadset()
    headset.run_in_thread()

    headset_control = HeadsetControl()

    feedback = HeadsetFeedback()

    headset_control.reset()

    # ========================================================
    # DEPTHAI
    # ========================================================

    pipeline = dai.Pipeline()

    cam_left = pipeline.create(
        dai.node.Camera
    ).build(
        dai.CameraBoardSocket.CAM_B
    )

    cam_right = pipeline.create(
        dai.node.Camera
    ).build(
        dai.CameraBoardSocket.CAM_C
    )

    left_img = cam_left.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.BGR888i,
        fps=FPS,
    )

    right_img = cam_right.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.BGR888i,
        fps=FPS,
    )

    q_left = left_img.createOutputQueue()
    q_right = right_img.createOutputQueue()

    pipeline.start()

    time.sleep(0.5)

    # ========================================================
    # LOAD URDF
    # ========================================================

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    print("\nLINK NAMES")
    for i, name in enumerate(robot.links.names):
        print(i, name)

    print("\n==============================")
    print("JOINT NAMES")
    print("==============================")

    for i, name in enumerate(robot.joints.names):
        print(i, name)

    print("\nActuated joints:")
    print(robot.joints.actuated_names)

    # ========================================================
    # IK TARGET LINKS
    # ========================================================

    target_link_names = [
        "leftgripper_base",
        "rightgripper_base",
        "middlecamera_cover",
    ]

    # ========================================================
    # VISER SERVER
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
    # DEBUG FRAMES
    # ========================================================

    head_frame = server.scene.add_frame(
        "/vr/head",
        axes_length=0.15,
        axes_radius=0.01,
    )

    left_controller_frame = server.scene.add_frame(
        "/vr/left_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    right_controller_frame = server.scene.add_frame(
        "/vr/right_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    left_target_frame = server.scene.add_frame(
        "/targets/left",
        axes_length=0.12,
        axes_radius=0.008,
    )

    right_target_frame = server.scene.add_frame(
        "/targets/right",
        axes_length=0.12,
        axes_radius=0.008,
    )

    middle_target_frame = server.scene.add_frame(
        "/targets/middle",
        axes_length=0.12,
        axes_radius=0.008,
    )

    left_ee_frame = server.scene.add_frame(
        "/ee/left",
        axes_length=0.10,
        axes_radius=0.006,
    )

    right_ee_frame = server.scene.add_frame(
        "/ee/right",
        axes_length=0.10,
        axes_radius=0.006,
    )

    middle_ee_frame = server.scene.add_frame(
        "/ee/middle",
        axes_length=0.10,
        axes_radius=0.006,
    )

    # ========================================================
    # TIMING GUI
    # ========================================================

    timing_handle = server.gui.add_number(
        "Elapsed (ms)",
        initial_value=0.0,
        disabled=True,
    )

    # ========================================================
    # ROBOT STATE
    # ========================================================

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    q_rest = q.copy()

    print("\n==============================")
    print("\nACTUATED JOINTS")
    print("\n==============================")
    for i, name in enumerate(
        robot.joints.actuated_names
    ):
        print(i, name)

    # ========================================================
    # NOMINAL TARGET POSES
    # ========================================================

    left_robot_pose = np.array([
        0.30,
        -0.20,
        0.25,
    ])

    right_robot_pose = np.array([
        0.35,
        0.25,
        0.35,
    ])

    middle_robot_pose = np.array([
        0.25,
        0.00,
        0.60,
    ])

    # ========================================================
    # TELEOP STATE
    # ========================================================

    teleop_active = False

    T_left_ref = None
    T_right_ref = None
    T_head_ref = None

    left_robot_ref = None
    right_robot_ref = None
    middle_robot_ref = None

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while pipeline.isRunning():

        start_time = time.time()

        # ----------------------------------------------------
        # CAMERA STREAMING
        # ----------------------------------------------------

        # left_img = q_left.get().getCvFrame()
        # right_img = q_right.get().getCvFrame()

        headset.send_images(
            left_img,
            right_img,
        )

        # ----------------------------------------------------
        # HEADSET DATA
        # ----------------------------------------------------

        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        # ----------------------------------------------------
        # BUILD CONTROLLER TRANSFORMS
        # ----------------------------------------------------

        T_head = pose2mat(
            headset_data.h_pos,
            headset_data.h_quat,
        )

        T_left = pose2mat(
            headset_data.l_pos,
            headset_data.l_quat,
        )

        T_right = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        head_frame.position = headset_data.h_pos
        head_frame.wxyz = xyzw_to_wxyz(
            headset_data.h_quat
        )

        left_controller_frame.position = headset_data.l_pos
        left_controller_frame.wxyz = xyzw_to_wxyz(
            headset_data.l_quat
        )

        right_controller_frame.position = headset_data.r_pos
        right_controller_frame.wxyz = xyzw_to_wxyz(
            headset_data.r_quat
        )

        # ----------------------------------------------------
        # SAFETY BUTTON
        # ----------------------------------------------------

        button_pressed = (
            headset_data.r_button_one
        )

        # ----------------------------------------------------
        # TELEOP START
        # ----------------------------------------------------

        if (
            button_pressed
            and not teleop_active
        ):

            teleop_active = True

            T_left_ref = T_left.copy()
            T_right_ref = T_right.copy()
            T_head_ref = T_head.copy()

            left_robot_ref = (
                left_robot_pose.copy()
            )

            right_robot_ref = (
                right_robot_pose.copy()
            )

            middle_robot_ref = (
                middle_robot_pose.copy()
            )

            print("Teleop ENABLED")

        # ----------------------------------------------------
        # TELEOP STOP
        # ----------------------------------------------------

        elif (
            not button_pressed
            and teleop_active
        ):

            teleop_active = False

            print("Teleop DISABLED")

        # ----------------------------------------------------
        # RUN TELEOP
        # ----------------------------------------------------

        if teleop_active:

            # ------------------------------------------------
            # Relative controller motion
            # ------------------------------------------------

            R_head = T_head[:3, :3]
            R_head_ref = T_head_ref[:3, :3]

            p_left_local = (
                R_head.T @
                (T_left[:3,3] - T_head[:3,3])
            )

            p_right_local = (
                R_head.T @
                (T_right[:3,3] - T_head[:3,3])
            )

            p_left_local_ref = (
                R_head_ref.T @
                (T_left_ref[:3,3] - T_head_ref[:3,3])
            )

            p_right_local_ref = (
                R_head_ref.T @
                (T_right_ref[:3,3] - T_head_ref[:3,3])
            )

            left_delta = p_left_local - p_left_local_ref
            right_delta = p_right_local - p_right_local_ref
            head_delta = np.zeros(3) #R_head_ref.T @ head_world_delta

            left_delta = np.array([
                -left_delta[2],   # VR forward -> robot x
                -left_delta[0],   # VR right   -> robot y
                left_delta[1],   # VR up      -> robot z
            ])

            right_delta = np.array([
                -right_delta[2],
                -right_delta[0],
                right_delta[1],
            ])

            head_delta = np.array([
                -head_delta[2],
                -head_delta[0],
                head_delta[1],
            ])

            # ------------------------------------------------
            # Apply relative motion
            # ------------------------------------------------

            left_robot_pose = (
                left_robot_ref + left_delta
            )

            right_robot_pose = (
                right_robot_ref + right_delta
            )

            middle_robot_pose = (
                middle_robot_ref + head_delta
            )

            # ------------------------------------------------
            # Build IK targets
            # ------------------------------------------------

            target_positions = np.array([
                left_robot_pose,
                right_robot_pose,
                middle_robot_pose,
            ])

            left_target_frame.position = left_robot_pose
            right_target_frame.position = right_robot_pose
            middle_target_frame.position = middle_robot_pose

            target_wxyzs = np.array([
                xyzw_to_wxyz(headset_data.l_quat),
                xyzw_to_wxyz(headset_data.r_quat),
                xyzw_to_wxyz(headset_data.h_quat),
            ])

            left_target_frame.wxyz = target_wxyzs[0]
            right_target_frame.wxyz = target_wxyzs[1]
            middle_target_frame.wxyz = target_wxyzs[2]

            q_prev = q.copy()

            # ------------------------------------------------
            # SOLVE IK
            # ------------------------------------------------

            q_new = (
                pks.solve_ik_with_multiple_targets(
                    robot,
                    target_link_names,
                    target_wxyzs,
                    target_positions,
                    q_prev=q_prev,
                    smoothness_weight=0.1,
                    rest_weight=0.05,
                    q_rest=q_rest,
                )
            )

            # ------------------------------------------------
            # Velocity limiting
            # ------------------------------------------------

            if q_new is not None:

                dt = max(
                    time.time() - start_time,
                    1e-4,
                )

                dq = q_new - q_prev

                dq = np.clip(
                    dq,
                    -V_MAX * dt,
                    V_MAX * dt,
                )

                q = q_prev + dq

        # ----------------------------------------------------
        # CRITICAL:
        # USE ACTUATED JOINTS ONLY
        # ----------------------------------------------------

        joint_dict = {
            name: value
            for name, value in zip(
                robot.joints.actuated_names,
                q
            )
        }

        urdf_vis.update_cfg(
            joint_dict
        )

        fk = robot.forward_kinematics(q)

        print(type(fk))
        print(fk.shape)
        print("BUTTON:", headset_data.r_button_one)

        T_left_ee = fk[8]
        T_right_ee = fk[18]
        T_middle_ee = fk[30]

        left_ee_frame.position = np.array(T_left_ee[:3])
        right_ee_frame.position = np.array(T_right_ee[:3])
        middle_ee_frame.position = np.array(T_middle_ee[:3])

        left_ee_frame.wxyz = np.array([
            T_left_ee[6],   # qw
            T_left_ee[3],   # qx
            T_left_ee[4],   # qy
            T_left_ee[5],   # qz
        ])

        right_ee_frame.wxyz = np.array([
            T_right_ee[6],
            T_right_ee[3],
            T_right_ee[4],
            T_right_ee[5],
        ])

        middle_ee_frame.wxyz = np.array([
            T_middle_ee[6],
            T_middle_ee[3],
            T_middle_ee[4],
            T_middle_ee[5],
        ])

        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        elapsed_ms = (
            time.time()
            - start_time
        ) * 1000.0

        timing_handle.value = (
            0.99 * timing_handle.value
            + 0.01 * elapsed_ms
        )

        # ----------------------------------------------------
        # HEADSET FEEDBACK
        # ----------------------------------------------------

        left_err = np.linalg.norm(
            left_robot_pose
            - np.array(T_left_ee[:3])
        )

        right_err = np.linalg.norm(
            right_robot_pose
            - np.array(T_right_ee[:3])
        )

        feedback.info = (
            f"TELEOP: "
            f"{'ON' if teleop_active else 'OFF'}\n\n"

            f"LEFT ERR: {left_err:.3f}\n"
            f"RIGHT ERR: {right_err:.3f}\n\n"

            f"LEFT TARGET\n"
            f"x: {left_robot_pose[0]:.3f}\n"
            f"y: {left_robot_pose[1]:.3f}\n"
            f"z: {left_robot_pose[2]:.3f}\n\n"

            f"RIGHT TARGET\n"
            f"x: {right_robot_pose[0]:.3f}\n"
            f"y: {right_robot_pose[1]:.3f}\n"
            f"z: {right_robot_pose[2]:.3f}"
        )

        headset.send_feedback(
            feedback
        )

        time.sleep(0.01)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
