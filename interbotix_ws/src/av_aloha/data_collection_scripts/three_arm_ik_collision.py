"""Coupled three-arm IK with manipulability and collision costs.

This experimental solver is intended for the Viser playground first. It adds:

- Three simultaneous pose costs.
- Per-arm translational manipulability costs.
- Whole-robot self-collision cost.
- Collision cost against one movable world geometry.
- Joint-limit constraint.
- Previous-command regularization and a hard velocity clamp.

The public wrapper keeps all array shapes fixed so changing GUI weights does
not trigger a new JAX compilation.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from pyroki.collision import CollGeom, RobotCollision


EPS = 1e-6
NUM_TARGETS = 3


@jaxls.Cost.create_factory
def previous_configuration_residual_scaled(
    vals,
    joint_var,
    prev_q,
    smoothness_scales,
):
    q = vals[joint_var]
    return smoothness_scales * (q - prev_q)


def _array(
    name: str,
    value,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}; got {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values.")
    return result


def make_three_arm_collision_ik_solver(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    left_target_link_name: str,
    right_target_link_name: str,
    middle_target_link_name: str,
) -> Tuple[Callable[..., np.ndarray], Callable[..., None], np.ndarray]:
    target_names = (
        left_target_link_name,
        right_target_link_name,
        middle_target_link_name,
    )
    target_indices_np = np.asarray(
        [robot.links.names.index(name) for name in target_names],
        dtype=np.int32,
    )
    target_indices_jax = jnp.asarray(target_indices_np)

    @jdc.jit
    def _solve_jax(
        prev_q: jax.Array,
        target_positions: jax.Array,
        target_wxyzs: jax.Array,
        dt: jax.Array,
        joint_velocity_limits: jax.Array,
        position_weights: jax.Array,
        orientation_weights: jax.Array,
        active_mask: jax.Array,
        manipulability_weights: jax.Array,
        dq_weight: jax.Array,
        self_collision_weight: jax.Array,
        world_collision_weight: jax.Array,
        collision_margin: jax.Array,
        world_obstacle: CollGeom,
    ) -> jax.Array:
        joint_var = robot.joint_var_cls(0)
        costs = []

        for target_idx in range(NUM_TARGETS):
            target_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyzs[target_idx]),
                target_positions[target_idx],
            )
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    target_pose,
                    target_indices_jax[target_idx],
                    pos_weight=(
                        position_weights[target_idx]
                        * active_mask[target_idx]
                    ),
                    ori_weight=(
                        orientation_weights[target_idx]
                        * active_mask[target_idx]
                    ),
                )
            )

            # A scalar target index gives one residual for one arm, allowing
            # independent weights for left, right, and middle.
            costs.append(
                pk.costs.manipulability_cost(
                    robot=robot,
                    joint_var=joint_var,
                    target_link_indices=target_indices_jax[target_idx],
                    weight=(
                        manipulability_weights[target_idx]
                        * active_mask[target_idx]
                    ),
                )
            )

        costs.append(pk.costs.limit_constraint(robot, joint_var))

        costs.append(
            pk.costs.self_collision_cost(
                robot=robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                margin=collision_margin,
                weight=self_collision_weight,
            )
        )

        costs.append(
            pk.costs.world_collision_cost(
                robot=robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                world_geom=world_obstacle,
                margin=collision_margin,
                weight=world_collision_weight,
            )
        )

        max_dq = jnp.maximum(joint_velocity_limits * dt, EPS)
        costs.append(
            previous_configuration_residual_scaled(
                joint_var=joint_var,
                prev_q=prev_q,
                smoothness_scales=dq_weight / max_dq,
            )
        )

        problem = jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[joint_var],
        )
        solution = problem.analyze().solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(
                lambda_initial=1.0,
            ),
        )
        return solution[joint_var]

    def solve(
        *,
        left_target_position,
        right_target_position,
        middle_target_position,
        left_target_wxyz,
        right_target_wxyz,
        middle_target_wxyz,
        prev_q,
        dt: float,
        joint_velocity_limits,
        world_obstacle: CollGeom,
        position_weights=(50.0, 50.0, 40.0),
        orientation_weights=(1.0, 1.0, 0.5),
        active_mask=(1.0, 1.0, 1.0),
        manipulability_weights=(0.0, 0.0, 0.0),
        dq_weight: float = 0.5,
        self_collision_weight: float = 0.0,
        world_collision_weight: float = 0.0,
        collision_margin: float = 0.03,
        block_until_ready: bool = True,
    ) -> np.ndarray:
        num_joints = robot.joints.num_actuated_joints

        positions = np.stack(
            [
                _array("left_target_position", left_target_position, (3,)),
                _array("right_target_position", right_target_position, (3,)),
                _array("middle_target_position", middle_target_position, (3,)),
            ]
        )
        quaternions = np.stack(
            [
                _array("left_target_wxyz", left_target_wxyz, (4,)),
                _array("right_target_wxyz", right_target_wxyz, (4,)),
                _array("middle_target_wxyz", middle_target_wxyz, (4,)),
            ]
        )
        prev_q_arr = _array("prev_q", prev_q, (num_joints,))
        limits = _array(
            "joint_velocity_limits",
            joint_velocity_limits,
            (num_joints,),
        )
        pos_weights = _array(
            "position_weights",
            position_weights,
            (NUM_TARGETS,),
        )
        ori_weights = _array(
            "orientation_weights",
            orientation_weights,
            (NUM_TARGETS,),
        )
        mask = _array("active_mask", active_mask, (NUM_TARGETS,))
        manip_weights = _array(
            "manipulability_weights",
            manipulability_weights,
            (NUM_TARGETS,),
        )

        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if np.any(limits <= 0.0):
            raise ValueError("Joint velocity limits must be positive.")
        if collision_margin < 0.0:
            raise ValueError("collision_margin cannot be negative.")

        q_next = _solve_jax(
            prev_q=jnp.asarray(prev_q_arr),
            target_positions=jnp.asarray(positions),
            target_wxyzs=jnp.asarray(quaternions),
            dt=jnp.asarray(dt, dtype=jnp.float32),
            joint_velocity_limits=jnp.asarray(limits),
            position_weights=jnp.asarray(pos_weights),
            orientation_weights=jnp.asarray(ori_weights),
            active_mask=jnp.asarray(mask),
            manipulability_weights=jnp.asarray(manip_weights),
            dq_weight=jnp.asarray(dq_weight, dtype=jnp.float32),
            self_collision_weight=jnp.asarray(
                self_collision_weight,
                dtype=jnp.float32,
            ),
            world_collision_weight=jnp.asarray(
                world_collision_weight,
                dtype=jnp.float32,
            ),
            collision_margin=jnp.asarray(
                collision_margin,
                dtype=jnp.float32,
            ),
            world_obstacle=world_obstacle,
        )

        if block_until_ready:
            q_next = q_next.block_until_ready()

        q_next_np = np.asarray(q_next, dtype=np.float32)
        max_dq = limits * np.float32(dt)
        dq = np.clip(q_next_np - prev_q_arr, -max_dq, max_dq)
        return prev_q_arr + dq

    def warmup(
        *,
        prev_q,
        joint_velocity_limits,
        dt: float,
        world_obstacle: CollGeom,
        position_weights=(50.0, 50.0, 40.0),
        orientation_weights=(1.0, 1.0, 0.5),
        active_mask=(1.0, 1.0, 1.0),
        manipulability_weights=(0.0, 0.0, 0.0),
        dq_weight: float = 0.5,
        self_collision_weight: float = 0.0,
        world_collision_weight: float = 0.0,
        collision_margin: float = 0.03,
    ) -> None:
        prev_q_arr = np.asarray(prev_q, dtype=np.float32)
        fk = robot.forward_kinematics(prev_q_arr)

        positions = []
        wxyzs = []
        for link_index in target_indices_np:
            pose = jaxlie.SE3(fk[link_index])
            positions.append(np.asarray(pose.translation(), dtype=np.float32))
            wxyzs.append(np.asarray(pose.rotation().wxyz, dtype=np.float32))

        solve(
            left_target_position=positions[0],
            right_target_position=positions[1],
            middle_target_position=positions[2],
            left_target_wxyz=wxyzs[0],
            right_target_wxyz=wxyzs[1],
            middle_target_wxyz=wxyzs[2],
            prev_q=prev_q_arr,
            dt=dt,
            joint_velocity_limits=joint_velocity_limits,
            world_obstacle=world_obstacle,
            position_weights=position_weights,
            orientation_weights=orientation_weights,
            active_mask=active_mask,
            manipulability_weights=manipulability_weights,
            dq_weight=dq_weight,
            self_collision_weight=self_collision_weight,
            world_collision_weight=world_collision_weight,
            collision_margin=collision_margin,
            block_until_ready=True,
        )

    return solve, warmup, target_indices_np
