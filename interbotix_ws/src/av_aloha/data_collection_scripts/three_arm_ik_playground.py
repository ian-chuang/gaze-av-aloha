"""Interactive coupled three-arm IK visualization for GIAVA."""

from __future__ import annotations

import time

import jaxlie
import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

from three_arm_ik import make_three_arm_ik_solver


URDF_PATH = "/home/devi/giava/giava.urdf"

LEFT_EE_LINK = "leftgripper_base"
RIGHT_EE_LINK = "rightgripper_base"
MIDDLE_EE_LINK = "middlecamera_cover"

DT = 0.05


def load_robot_and_urdf() -> tuple[URDF, pk.Robot]:
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)
    return urdf, robot


def get_link_pose(
    fk,
    link_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    T_world_link = jaxlie.SE3(fk[link_index])
    position = np.asarray(
        T_world_link.translation(),
        dtype=np.float32,
    )
    wxyz = np.asarray(
        T_world_link.rotation().wxyz,
        dtype=np.float32,
    )
    return position, wxyz


def main() -> None:
    urdf, robot = load_robot_and_urdf()

    link_names = (
        LEFT_EE_LINK,
        RIGHT_EE_LINK,
        MIDDLE_EE_LINK,
    )
    link_indices = {
        name: robot.links.names.index(name)
        for name in link_names
    }

    print("Actuated joints:", robot.joints.num_actuated_joints)
    print("Left target index:", link_indices[LEFT_EE_LINK])
    print("Right target index:", link_indices[RIGHT_EE_LINK])
    print("Middle target index:", link_indices[MIDDLE_EE_LINK])

    cfg = np.asarray(
        robot.joint_var_cls(0).default_factory(),
        dtype=np.float32,
    )

    solve_three_arm, warmup_three_arm, _ = make_three_arm_ik_solver(
        robot=robot,
        left_target_link_name=LEFT_EE_LINK,
        right_target_link_name=RIGHT_EE_LINK,
        middle_target_link_name=MIDDLE_EE_LINK,
    )

    joint_velocity_limits = np.full(
        robot.joints.num_actuated_joints,
        2.0,
        dtype=np.float32,
    )

    fk = robot.forward_kinematics(cfg)

    left_position, left_wxyz = get_link_pose(
        fk,
        link_indices[LEFT_EE_LINK],
    )
    right_position, right_wxyz = get_link_pose(
        fk,
        link_indices[RIGHT_EE_LINK],
    )
    middle_position, middle_wxyz = get_link_pose(
        fk,
        link_indices[MIDDLE_EE_LINK],
    )

    server = viser.ViserServer()
    server.scene.add_grid(
        "/ground",
        width=2,
        height=2,
        cell_size=0.1,
    )

    urdf_vis = ViserUrdf(
        server,
        urdf,
        root_node_name="/robot",
    )
    urdf_vis.update_cfg(cfg)

    left_target = server.scene.add_transform_controls(
        "/targets/left",
        scale=0.15,
        position=tuple(left_position),
        wxyz=tuple(left_wxyz),
    )
    right_target = server.scene.add_transform_controls(
        "/targets/right",
        scale=0.15,
        position=tuple(right_position),
        wxyz=tuple(right_wxyz),
    )
    middle_target = server.scene.add_transform_controls(
        "/targets/middle",
        scale=0.15,
        position=tuple(middle_position),
        wxyz=tuple(middle_wxyz),
    )

    server.gui.add_markdown(
        """
### Three-arm coupled IK

All enabled targets are solved in one optimization. Disable an arm to remove
its pose cost while retaining previous-configuration regularization.
"""
    )

    left_active_handle = server.gui.add_checkbox(
        "Left target active",
        True,
    )
    right_active_handle = server.gui.add_checkbox(
        "Right target active",
        True,
    )
    middle_active_handle = server.gui.add_checkbox(
        "Middle target active",
        True,
    )

    left_pos_weight_handle = server.gui.add_slider(
        "Left position weight",
        0.0,
        100.0,
        0.5,
        50.0,
    )
    right_pos_weight_handle = server.gui.add_slider(
        "Right position weight",
        0.0,
        100.0,
        0.5,
        50.0,
    )
    middle_pos_weight_handle = server.gui.add_slider(
        "Middle position weight",
        0.0,
        100.0,
        0.5,
        50.0,
    )

    left_ori_weight_handle = server.gui.add_slider(
        "Left orientation weight",
        0.0,
        20.0,
        0.1,
        2.0,
    )
    right_ori_weight_handle = server.gui.add_slider(
        "Right orientation weight",
        0.0,
        20.0,
        0.1,
        2.0,
    )
    middle_ori_weight_handle = server.gui.add_slider(
        "Middle orientation weight",
        0.0,
        20.0,
        0.1,
        2.0,
    )

    dq_weight_handle = server.gui.add_slider(
        "Previous-q weight",
        0.0,
        5.0,
        0.01,
        0.5,
    )
    velocity_limit_handle = server.gui.add_slider(
        "Joint velocity limit (rad/s)",
        0.1,
        4.0,
        0.1,
        2.0,
    )

    left_manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability/left",
        target_link_name=LEFT_EE_LINK,
    )
    right_manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability/right",
        target_link_name=RIGHT_EE_LINK,
    )
    middle_manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability/middle",
        target_link_name=MIDDLE_EE_LINK,
    )

    show_manip_handle = server.gui.add_checkbox(
        "Show manipulability ellipsoids",
        True,
    )
    timing_handle = server.gui.add_number(
        "Solve time (ms)",
        0.0,
        disabled=True,
    )
    left_position_error_handle = server.gui.add_number(
        "Left position error (m)",
        0.0,
        disabled=True,
    )
    right_position_error_handle = server.gui.add_number(
        "Right position error (m)",
        0.0,
        disabled=True,
    )
    middle_position_error_handle = server.gui.add_number(
        "Middle position error (m)",
        0.0,
        disabled=True,
    )

    warmup_three_arm(
        prev_q=cfg,
        joint_velocity_limits=joint_velocity_limits,
        dt=DT,
        position_weights=np.asarray(
            [
                left_pos_weight_handle.value,
                right_pos_weight_handle.value,
                middle_pos_weight_handle.value,
            ],
            dtype=np.float32,
        ),
        orientation_weights=np.asarray(
            [
                left_ori_weight_handle.value,
                right_ori_weight_handle.value,
                middle_ori_weight_handle.value,
            ],
            dtype=np.float32,
        ),
        active_mask=np.ones(3, dtype=np.float32),
        dq_weight=dq_weight_handle.value,
    )

    print("Three-arm solver compiled.")
    print("Open the Viser URL printed above.")

    try:
        while True:
            start_time = time.perf_counter()

            joint_velocity_limits.fill(
                np.float32(velocity_limit_handle.value)
            )

            position_weights = np.asarray(
                [
                    left_pos_weight_handle.value,
                    right_pos_weight_handle.value,
                    middle_pos_weight_handle.value,
                ],
                dtype=np.float32,
            )
            orientation_weights = np.asarray(
                [
                    left_ori_weight_handle.value,
                    right_ori_weight_handle.value,
                    middle_ori_weight_handle.value,
                ],
                dtype=np.float32,
            )
            active_mask = np.asarray(
                [
                    float(left_active_handle.value),
                    float(right_active_handle.value),
                    float(middle_active_handle.value),
                ],
                dtype=np.float32,
            )

            cfg = solve_three_arm(
                left_target_position=np.asarray(
                    left_target.position,
                    dtype=np.float32,
                ),
                right_target_position=np.asarray(
                    right_target.position,
                    dtype=np.float32,
                ),
                middle_target_position=np.asarray(
                    middle_target.position,
                    dtype=np.float32,
                ),
                left_target_wxyz=np.asarray(
                    left_target.wxyz,
                    dtype=np.float32,
                ),
                right_target_wxyz=np.asarray(
                    right_target.wxyz,
                    dtype=np.float32,
                ),
                middle_target_wxyz=np.asarray(
                    middle_target.wxyz,
                    dtype=np.float32,
                ),
                prev_q=cfg,
                dt=DT,
                joint_velocity_limits=joint_velocity_limits,
                position_weights=position_weights,
                orientation_weights=orientation_weights,
                active_mask=active_mask,
                dq_weight=dq_weight_handle.value,
                block_until_ready=True,
            )

            urdf_vis.update_cfg(cfg)

            show_manip = show_manip_handle.value
            left_manip_ellipse.set_visibility(show_manip)
            right_manip_ellipse.set_visibility(show_manip)
            middle_manip_ellipse.set_visibility(show_manip)

            if show_manip:
                left_manip_ellipse.update(cfg)
                right_manip_ellipse.update(cfg)
                middle_manip_ellipse.update(cfg)

            fk_solution = robot.forward_kinematics(cfg)
            solved_left_position, _ = get_link_pose(
                fk_solution,
                link_indices[LEFT_EE_LINK],
            )
            solved_right_position, _ = get_link_pose(
                fk_solution,
                link_indices[RIGHT_EE_LINK],
            )
            solved_middle_position, _ = get_link_pose(
                fk_solution,
                link_indices[MIDDLE_EE_LINK],
            )

            left_position_error_handle.value = float(
                np.linalg.norm(
                    solved_left_position
                    - np.asarray(left_target.position)
                )
            )
            right_position_error_handle.value = float(
                np.linalg.norm(
                    solved_right_position
                    - np.asarray(right_target.position)
                )
            )
            middle_position_error_handle.value = float(
                np.linalg.norm(
                    solved_middle_position
                    - np.asarray(middle_target.position)
                )
            )

            elapsed = time.perf_counter() - start_time
            timing_handle.value = elapsed * 1000.0
            time.sleep(max(0.0, DT - elapsed))
    except KeyboardInterrupt:
        print("\nExiting three-arm IK playground.")


if __name__ == "__main__":
    main()
