"""Bimanual IK

Same as 01_basic_ik.py, but with two end effectors!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from yourdfpy import URDF
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf

import sys

sys.path.append("/home/devi/giava/pyroki/examples")

from pyroki_snippets._solve_ik_with_multiple_targets import (
    solve_ik_with_multiple_targets,
)
import pyroki_snippets as pks

v_max = 2
    

def main():
    urdf = URDF.load("/home/devi/giava/giava.urdf")
    target_link_names = [
        "leftgripper_base",
        "rightgripper_base",
        "middlecamera_cover"
    ]

    robot = pk.Robot.from_urdf(urdf)

    print(robot.joints.names)

    '''
    ('base_leftbase_link_fixed', 'leftwaist', 'leftshoulder', 'leftelbow', 'leftforearm_roll', 'leftwrist_angle', 
    'leftwrist_rotate', 'leftgripper_link_leftgripper_base_fixed', 'leftleft_finger', 'leftright_finger', 
    'base_rightbase_link_fixed', 'rightwaist', 'rightshoulder', 'rightelbow', 'rightforearm_roll', 'rightwrist_angle', 
    'rightwrist_rotate', 'rightgripper_link_rightgripper_base_fixed', 'rightleft_finger', 'rightright_finger', 
    'middlebase_link', 'middleshoulder_link', 'middleupper_arm_link', 'middleupper_forearm_link', 
    'middlelower_forearm_link', 'middlewrist_link', 'middlepan_link', 'base_middlebase_link_fixed', 
    'middlecamera_body_fixed', 'middlecamera_cover_fixed')
    '''

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # target_wxyzs = np.array([
    #     left_wxyz,
    #     right_wxyz,
    # ])

    # target_positions = np.array([
    #     left_pos,
    #     right_pos,
    # ])

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    q = np.zeros(robot.joints.num_actuated_joints)

    # # --- GUI button ---
    # save_button = server.gui.add_button("Save Current Pose")

    # @save_button.on_click
    # def _(_event):
    #     saved_q["value"] = q.copy()
    #     print("Saved q:", saved_q["value"])


    q_rest = np.zeros_like(q)  # or better: a natural pose

    q = np.array([
        0.2, 0.2, 0.2,  0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2,  0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2,  0.2, 1.2, 0.2, 0.2,  0.2,  0.2])

    while True:
        start_time = time.time()

        target_wxyzs = np.array([
            ik_target_0.wxyz,
            ik_target_1.wxyz,
            ik_target_2.wxyz
        ])

        target_positions = np.array([
            ik_target_0.position,
            ik_target_1.position,
            ik_target_2.position
        ])

        q_prev = q.copy()

        q_new = pks.solve_ik_with_multiple_targets(
            robot,
            target_link_names,
            target_wxyzs,
            target_positions,
            q_prev=q,
            smoothness_weight=0.1,
            rest_weight=0.05,
            q_rest=q_rest
        )

        dt = time.time() - start_time

        if q_new is not None:
            dq = q_new - q_prev
            # dq = np.clip(dq, -v_max * dt, v_max * dt)
            q = q_prev + dq

        if q is None:
            q = np.zeros(robot.joints.num_actuated_joints)

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(q)

if __name__ == "__main__":
    main()
