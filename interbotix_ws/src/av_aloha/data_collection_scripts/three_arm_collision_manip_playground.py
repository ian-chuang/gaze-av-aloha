"""Three-arm IK playground with real collision and manipulability costs."""

from __future__ import annotations

import time

import jaxlie
import numpy as np
import pyroki as pk
import viser
from pyroki.collision import RobotCollision, Sphere
from viser.extras import ViserUrdf
from yourdfpy import URDF

from three_arm_ik_collision import make_three_arm_collision_ik_solver


URDF_PATH = "/home/devi/giava/giava.urdf"
LEFT_EE_LINK = "leftgripper_base"
RIGHT_EE_LINK = "rightgripper_base"
MIDDLE_EE_LINK = "middlecamera_cover"
DT = 0.05


def link_pose(fk, index: int) -> tuple[np.ndarray, np.ndarray]:
    pose = jaxlie.SE3(fk[index])
    return (
        np.asarray(pose.translation(), dtype=np.float32),
        np.asarray(pose.rotation().wxyz, dtype=np.float32),
    )


def main() -> None:
    urdf = URDF.load(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    indices = {
        "left": robot.links.names.index(LEFT_EE_LINK),
        "right": robot.links.names.index(RIGHT_EE_LINK),
        "middle": robot.links.names.index(MIDDLE_EE_LINK),
    }

    cfg = np.asarray(
        robot.joint_var_cls(0).default_factory(),
        dtype=np.float32,
    )
    velocity_limits = np.full(
        robot.joints.num_actuated_joints,
        2.0,
        dtype=np.float32,
    )

    fk = robot.forward_kinematics(cfg)
    left_pos, left_wxyz = link_pose(fk, indices["left"])
    right_pos, right_wxyz = link_pose(fk, indices["right"])
    middle_pos, middle_wxyz = link_pose(fk, indices["middle"])

    solve, warmup, _ = make_three_arm_collision_ik_solver(
        robot=robot,
        robot_coll=robot_coll,
        left_target_link_name=LEFT_EE_LINK,
        right_target_link_name=RIGHT_EE_LINK,
        middle_target_link_name=MIDDLE_EE_LINK,
    )

    obstacle_radius = 0.10
    obstacle_local = Sphere.from_center_and_radius(
        np.zeros(3, dtype=np.float32),
        np.asarray([obstacle_radius], dtype=np.float32),
    )

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(cfg)

    left_target = server.scene.add_transform_controls(
        "/targets/left",
        scale=0.15,
        position=tuple(left_pos),
        wxyz=tuple(left_wxyz),
    )
    right_target = server.scene.add_transform_controls(
        "/targets/right",
        scale=0.15,
        position=tuple(right_pos),
        wxyz=tuple(right_wxyz),
    )
    middle_target = server.scene.add_transform_controls(
        "/targets/middle",
        scale=0.15,
        position=tuple(middle_pos),
        wxyz=tuple(middle_wxyz),
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

    left_pos_w = server.gui.add_slider(
        "Left position weight", 0.0, 100.0, 0.5, 50.0
    )
    right_pos_w = server.gui.add_slider(
        "Right position weight", 0.0, 100.0, 0.5, 50.0
    )
    middle_pos_w = server.gui.add_slider(
        "Middle position weight", 0.0, 100.0, 0.5, 40.0
    )

    left_ori_w = server.gui.add_slider(
        "Left orientation weight", 0.0, 20.0, 0.1, 1.0
    )
    right_ori_w = server.gui.add_slider(
        "Right orientation weight", 0.0, 20.0, 0.1, 1.0
    )
    middle_ori_w = server.gui.add_slider(
        "Middle orientation weight", 0.0, 20.0, 0.1, 0.5
    )

    left_manip_w = server.gui.add_slider(
        "Left manipulability weight", 0.0, 2.0, 0.001, 0.0
    )
    right_manip_w = server.gui.add_slider(
        "Right manipulability weight", 0.0, 2.0, 0.001, 0.0
    )
    middle_manip_w = server.gui.add_slider(
        "Middle manipulability weight", 0.0, 2.0, 0.001, 0.0
    )

    dq_w = server.gui.add_slider(
        "Previous-q weight", 0.0, 5.0, 0.01, 0.5
    )
    self_coll_w = server.gui.add_slider(
        "Self-collision weight", 0.0, 100.0, 0.5, 0.0
    )
    world_coll_w = server.gui.add_slider(
        "World-collision weight", 0.0, 100.0, 0.5, 0.0
    )
    margin = server.gui.add_slider(
        "Collision margin (m)", 0.0, 0.15, 0.005, 0.03
    )
    velocity = server.gui.add_slider(
        "Joint velocity limit (rad/s)", 0.1, 4.0, 0.1, 2.0
    )

    left_manip = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability/left",
        target_link_name=LEFT_EE_LINK,
    )
    right_manip = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability/right",
        target_link_name=RIGHT_EE_LINK,
    )
    middle_manip = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability/middle",
        target_link_name=MIDDLE_EE_LINK,
    )

    show_manip = server.gui.add_checkbox(
        "Show manipulability ellipsoids", True
    )
    solve_time = server.gui.add_number(
        "Solve time (ms)", 0.0, disabled=True
    )
    left_value = server.gui.add_number(
        "Left Yoshikawa index", 0.0, disabled=True
    )
    right_value = server.gui.add_number(
        "Right Yoshikawa index", 0.0, disabled=True
    )
    middle_value = server.gui.add_number(
        "Middle Yoshikawa index", 0.0, disabled=True
    )

    world_obstacle = obstacle_local.transform_from_wxyz_position(
        wxyz=np.asarray(obstacle_handle.wxyz, dtype=np.float32),
        position=np.asarray(obstacle_handle.position, dtype=np.float32),
    )

    print("Compiling collision-aware three-arm solver...")
    warmup(
        prev_q=cfg,
        joint_velocity_limits=velocity_limits,
        dt=DT,
        world_obstacle=world_obstacle,
    )
    print("Solver ready.")

    try:
        while True:
            started = time.perf_counter()

            velocity_limits.fill(np.float32(velocity.value))
            world_obstacle = obstacle_local.transform_from_wxyz_position(
                wxyz=np.asarray(
                    obstacle_handle.wxyz,
                    dtype=np.float32,
                ),
                position=np.asarray(
                    obstacle_handle.position,
                    dtype=np.float32,
                ),
            )

            cfg = solve(
                left_target_position=np.asarray(
                    left_target.position, dtype=np.float32
                ),
                right_target_position=np.asarray(
                    right_target.position, dtype=np.float32
                ),
                middle_target_position=np.asarray(
                    middle_target.position, dtype=np.float32
                ),
                left_target_wxyz=np.asarray(
                    left_target.wxyz, dtype=np.float32
                ),
                right_target_wxyz=np.asarray(
                    right_target.wxyz, dtype=np.float32
                ),
                middle_target_wxyz=np.asarray(
                    middle_target.wxyz, dtype=np.float32
                ),
                prev_q=cfg,
                dt=DT,
                joint_velocity_limits=velocity_limits,
                world_obstacle=world_obstacle,
                position_weights=np.asarray(
                    [left_pos_w.value, right_pos_w.value, middle_pos_w.value],
                    dtype=np.float32,
                ),
                orientation_weights=np.asarray(
                    [left_ori_w.value, right_ori_w.value, middle_ori_w.value],
                    dtype=np.float32,
                ),
                manipulability_weights=np.asarray(
                    [
                        left_manip_w.value,
                        right_manip_w.value,
                        middle_manip_w.value,
                    ],
                    dtype=np.float32,
                ),
                dq_weight=dq_w.value,
                self_collision_weight=self_coll_w.value,
                world_collision_weight=world_coll_w.value,
                collision_margin=margin.value,
                block_until_ready=True,
            )

            urdf_vis.update_cfg(cfg)

            visible = show_manip.value
            for ellipse in (left_manip, right_manip, middle_manip):
                ellipse.set_visibility(visible)
                if visible:
                    ellipse.update(cfg)

            if visible:
                left_value.value = float(left_manip.manipulability)
                right_value.value = float(right_manip.manipulability)
                middle_value.value = float(middle_manip.manipulability)

            elapsed = time.perf_counter() - started
            solve_time.value = elapsed * 1000.0
            time.sleep(max(0.0, DT - elapsed))
    except KeyboardInterrupt:
        print("\nExiting playground.")


if __name__ == "__main__":
    main()
