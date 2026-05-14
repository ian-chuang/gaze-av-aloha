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


# CONFIG

URDF_PATH = "/home/devi/giava/giava.urdf"

# RIGHT_EE_LINK = "rightgripper_base"

# LEFT_EE_LINK = "leftgripper_base"

CAMERA_EE_LINK = "middlecamera_body"

# MAIN

def main():

    # ROBOT

    urdf = URDF.load(URDF_PATH)

    robot = pk.Robot.from_urdf(urdf)

    q = np.zeros(
        robot.joints.num_actuated_joints
    )

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

    # TARGET FRAME

    target_frame = server.scene.add_frame(
        "/target",
        axes_length=0.15,
        axes_radius=0.01,
    )

    # POSITION SLIDERS

    x_slider = server.gui.add_slider(
        "x",
        min=-0.6,
        max=0.6,
        step=0.01,
        initial_value=-0.2,
    )

    y_slider = server.gui.add_slider(
        "y",
        min=-0.6,
        max=0.6,
        step=0.01,
        initial_value=0.0,
    )

    z_slider = server.gui.add_slider(
        "z",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=0.45,
    )

    # ROTATION SLIDERS

    rx_slider = server.gui.add_slider(
        "roll_x_deg",
        min=-180,
        max=180,
        step=1,
        initial_value=180,
    )

    ry_slider = server.gui.add_slider(
        "pitch_y_deg",
        min=-180,
        max=180,
        step=1,
        initial_value=0,
    )

    rz_slider = server.gui.add_slider(
        "yaw_z_deg",
        min=-180,
        max=180,
        step=1,
        initial_value=90,
    )

    # MAIN LOOP

    while True:

        # TARGET POSITION

        target_position = np.array([
            x_slider.value,
            y_slider.value,
            z_slider.value,
        ])

        # TARGET ROTATION

        rot = R.from_euler(
            "xyz",
            [
                rx_slider.value,
                ry_slider.value,
                rz_slider.value,
            ],
            degrees=True,
        )

        R_target = rot.as_matrix()

        quat_xyzw = rot.as_quat()

        target_wxyz = np.array([
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ])

        # VISUALIZE TARGET FRAME

        target_frame.position = target_position

        target_frame.wxyz = target_wxyz

        # SOLVE IK

        q_new = pks.solve_ik(
            robot=robot,
            target_link_name=CAMERA_EE_LINK,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )

        if q_new is not None:
            q = q_new

        # UPDATE ROBOT

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

        # DEBUG PRINT

        print(
            f"xyz = {target_position} | "
            f"rpy = {[rx_slider.value, ry_slider.value, rz_slider.value]}"
        )

        time.sleep(0.01)


# ENTRY

if __name__ == "__main__":
    main()