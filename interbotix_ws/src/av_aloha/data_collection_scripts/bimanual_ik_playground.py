"""Interactive bimanual IK visualization."""

import sys
import time

import jaxlie
import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

sys.path.append("/home/devi/giava/pyroki/examples")
import pyroki_snippets as pks
from pyroki.collision import RobotCollision, Sphere

LEFT_EE_LINK = "leftgripper_base"
RIGHT_EE_LINK = "rightgripper_base"

DT = 0.05


def load_robot_and_urdf():
    urdf = URDF.load("/home/devi/giava/giava.urdf")
    robot = pk.Robot.from_urdf(urdf)
    return urdf, robot


def main():
    urdf, robot = load_robot_and_urdf()
    robot_coll = RobotCollision.from_urdf(urdf)

    left_link_index = robot.links.names.index(LEFT_EE_LINK)
    right_link_index = robot.links.names.index(RIGHT_EE_LINK)

    print("Actuated joints:", robot.joints.num_actuated_joints)
    print("Left target index:", left_link_index)
    print("Right target index:", right_link_index)

    cfg = np.asarray(
        robot.joint_var_cls(0).default_factory(),
        dtype=np.float32,
    )

    solve_bimanual, warmup_bimanual, target_indices = (
        pks.make_bimanual_ik_solver(
            robot=robot,
            robot_coll=robot_coll,
            left_target_link_name=LEFT_EE_LINK,
            right_target_link_name=RIGHT_EE_LINK,
        )
    )

    joint_velocity_limits = np.full(
        robot.joints.num_actuated_joints,
        2.0,
        dtype=np.float32,
    )

    # Obtain initial targets from the current robot FK.
    fk = robot.forward_kinematics(cfg)

    T_world_left = jaxlie.SE3(fk[left_link_index])
    T_world_right = jaxlie.SE3(fk[right_link_index])

    left_position = np.asarray(T_world_left.translation())
    left_wxyz = np.asarray(T_world_left.rotation().wxyz)

    right_position = np.asarray(T_world_right.translation())
    right_wxyz = np.asarray(T_world_right.rotation().wxyz)

    obstacle_radius = 0.10

    obstacle_base = Sphere.from_center_and_radius(
        np.zeros(3, dtype=np.float32),
        np.array([obstacle_radius], dtype=np.float32),
    )

    server = viser.ViserServer()
    server.scene.add_grid(
        "/ground",
        width=2,
        height=2,
        cell_size=0.1,
    )

    obstacle_handle = server.scene.add_transform_controls(
        "/obstacle",
        scale=0.15,
        position=(0.35, 0.0, 0.35),
    )

    server.scene.add_icosphere(
        "/obstacle/mesh",
        radius=obstacle_radius,
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

    pos_weight_handle = server.gui.add_slider(
        "Position weight",
        0.0,
        100.0,
        0.5,
        50.0,
    )

    ori_weight_handle = server.gui.add_slider(
        "Orientation weight",
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

    manip_weight_handle = server.gui.add_slider(
        "Manipulability weight",
        0.0,
        5.0,
        0.01,
        0.0,
    )

    self_collision_weight_handle = server.gui.add_slider(
        "Self-collision weight",
        0.0,
        100.0,
        0.5,
        0.0,
    )

    collision_margin_handle = server.gui.add_slider(
        "Collision margin (m)",
        0.0,
        0.15,
        0.005,
        0.03,
    )

    world_collision_weight_handle = server.gui.add_slider(
        "World-collision weight",
        0.0,
        100.0,
        0.5,
        0.0,
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

    show_manip_handle = server.gui.add_checkbox(
        "Show manipulability ellipsoids",
        True,
    )

    timing_handle = server.gui.add_number(
        "Solve time (ms)",
        0.0,
        disabled=True,
    )

    world_obstacle = obstacle_base.transform_from_wxyz_position(
        wxyz=np.asarray(obstacle_handle.wxyz, dtype=np.float32),
        position=np.asarray(obstacle_handle.position, dtype=np.float32),
    )

    # Compile the solver once before interactive use.
    warmup_bimanual(
        prev_q=cfg,
        joint_velocity_limits=joint_velocity_limits,
        dt=DT,
        pos_weight=pos_weight_handle.value,
        ori_weight=ori_weight_handle.value,
        dq_weight=dq_weight_handle.value,
        manip_weight=manip_weight_handle.value,
        self_collision_weight=self_collision_weight_handle.value,
        world_collision_weight=world_collision_weight_handle.value,
        collision_margin=collision_margin_handle.value,
        world_obstacle=world_obstacle,
    )

    while True:
        start_time = time.perf_counter()

        world_obstacle = obstacle_base.transform_from_wxyz_position(
            wxyz=np.asarray(
                obstacle_handle.wxyz,
                dtype=np.float32,
            ),
            position=np.asarray(
                obstacle_handle.position,
                dtype=np.float32,
            ),
        )

        cfg = solve_bimanual(
            left_target_position=np.asarray(
                left_target.position,
                dtype=np.float32,
            ),
            right_target_position=np.asarray(
                right_target.position,
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
            prev_q=cfg,
            dt=DT,
            joint_velocity_limits=joint_velocity_limits,
            pos_weight=pos_weight_handle.value,
            ori_weight=ori_weight_handle.value,
            dq_weight=dq_weight_handle.value,
            manip_weight=manip_weight_handle.value,
            self_collision_weight=self_collision_weight_handle.value,
            world_collision_weight=world_collision_weight_handle.value,
            collision_margin=collision_margin_handle.value,
            world_obstacle=world_obstacle,
            block_until_ready=True,
        )

        urdf_vis.update_cfg(cfg)

        left_manip_ellipse.set_visibility(show_manip_handle.value)
        right_manip_ellipse.set_visibility(show_manip_handle.value)

        left_manip_ellipse.update(cfg)
        right_manip_ellipse.update(cfg)

        elapsed = time.perf_counter() - start_time
        timing_handle.value = elapsed * 1000.0

        time.sleep(max(0.0, DT - elapsed))

if __name__ == "__main__":
    main()