import time
import numpy as np
import viser

from viser.extras import ViserUrdf
from yourdfpy import URDF

import pyroki as pk
import sys

sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks

from webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl as HeadsetControl

from transform_utils import (
    pose2mat,
    xyzw_to_wxyz,
)

# ============================================================
# CONFIG
# ============================================================

URDF_PATH = "/home/devi/giava/giava.urdf"

RIGHT_EE_LINK = "rightgripper_base"
RIGHT_EE_INDEX = 8

POSITION_SCALE = 1.0

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
    headset_control.reset()

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
    # VISUALIZATION FRAMES
    # ========================================================

    controller_frame = server.scene.add_frame(
        "/vr/right_controller",
        axes_length=0.15,
        axes_radius=0.01,
    )

    target_frame = server.scene.add_frame(
        "/target/right",
        axes_length=0.12,
        axes_radius=0.008,
    )

    ee_frame = server.scene.add_frame(
        "/ee/right",
        axes_length=0.10,
        axes_radius=0.006,
    )

    # ========================================================
    # ROBOT STATE
    # ========================================================

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

    # ========================================================
    # INITIAL TARGET
    # ========================================================

    right_robot_pose = np.array([
        0.35,
        0.25,
        0.35,
    ])

    # ========================================================
    # TELEOP STATE
    # ========================================================

    teleop_active = False

    T_right_ref = None

    right_robot_ref = None

    # ========================================================
    # AXIS MAPPING
    # ========================================================

    R_vr_to_robot = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
    ])

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
        # BUILD TRANSFORM
        # ----------------------------------------------------

        T_right = pose2mat(
            headset_data.r_pos,
            headset_data.r_quat,
        )

        # ----------------------------------------------------
        # VISUALIZE CONTROLLER
        # ----------------------------------------------------

        controller_frame.position = headset_data.r_pos

        controller_frame.wxyz = xyzw_to_wxyz(
            headset_data.r_quat
        )

        # ----------------------------------------------------
        # BUTTON STATE
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

            T_right_ref = T_right.copy()

            right_robot_ref = (
                right_robot_pose.copy()
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
            # PURE CONTROLLER DELTA
            # ------------------------------------------------

            controller_delta = (
                T_right[:3,3]
                - T_right_ref[:3,3]
            )

            # ------------------------------------------------
            # MAP VR FRAME -> ROBOT FRAME
            # ------------------------------------------------

            delta_robot = (
                R_vr_to_robot
                @ controller_delta
            )

            delta_robot *= POSITION_SCALE

            # ------------------------------------------------
            # UPDATE TARGET
            # ------------------------------------------------

            right_robot_pose = (
                right_robot_ref
                + delta_robot
            )

        # ----------------------------------------------------
        # VISUALIZE TARGET
        # ----------------------------------------------------

        target_frame.position = (
            right_robot_pose
        )

        # ----------------------------------------------------
        # SOLVE IK
        # ----------------------------------------------------

        q_new = pks.solve_ik(
            robot=robot,
            target_link_name=RIGHT_EE_LINK,
            target_position=right_robot_pose,

            # fixed orientation
            target_wxyz=np.array([
                1.0,
                0.0,
                0.0,
                0.0,
            ]),
        )

        if q_new is not None:
            q = q_new

        # ----------------------------------------------------
        # UPDATE ROBOT VISUALIZATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # FORWARD KINEMATICS
        # ----------------------------------------------------

        fk = robot.forward_kinematics(q)

        T_right_ee = fk[RIGHT_EE_INDEX]

        # ----------------------------------------------------
        # VISUALIZE EE
        # ----------------------------------------------------

        ee_frame.position = np.array(
            T_right_ee[:3]
        )

        ee_frame.wxyz = np.array([
            T_right_ee[6],
            T_right_ee[3],
            T_right_ee[4],
            T_right_ee[5],
        ])

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        err = np.linalg.norm(
            right_robot_pose
            - np.array(T_right_ee[:3])
        )

        elapsed_ms = (
            time.time()
            - start_time
        ) * 1000.0

        print(
            f"teleop={teleop_active} "
            f"err={err:.3f} "
            f"dt={elapsed_ms:.1f}ms"
        )

        time.sleep(0.01)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()