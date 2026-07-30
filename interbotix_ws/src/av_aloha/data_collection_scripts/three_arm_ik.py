"""Coupled three-arm inverse kinematics for the GIAVA robot.

This module solves left, right, and middle arm targets in one least-squares
problem. It includes:

- Three simultaneous end-effector pose costs.
- Per-arm position and orientation weights.
- Per-arm active masks.
- Robot joint-limit constraints.
- Previous-configuration regularization scaled by velocity limits.
- A final hard joint-velocity clamp.

Collision and manipulability costs are intentionally not exposed here because
the original bimanual solver accepted those arguments without adding the
corresponding costs to the least-squares problem.
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


EPS = 1e-6
NUM_TARGETS = 3


@jaxls.Cost.create_factory
def previous_configuration_residual_scaled(
    vals,
    joint_var,
    prev_q,
    smoothness_scales,
):
    """Penalize changes from the previously commanded configuration."""
    q = vals[joint_var]
    return smoothness_scales * (q - prev_q)


def _validate_vector(
    name: str,
    value: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def make_three_arm_ik_solver(
    robot: pk.Robot,
    left_target_link_name: str,
    right_target_link_name: str,
    middle_target_link_name: str,
) -> Tuple[Callable[..., np.ndarray], Callable[..., None], np.ndarray]:
    """Create a compiled, coupled IK solver for all three arms.

    The order used everywhere is:

        0: left
        1: right
        2: middle
    """
    target_link_names = (
        left_target_link_name,
        right_target_link_name,
        middle_target_link_name,
    )

    missing_links = [
        link_name
        for link_name in target_link_names
        if link_name not in robot.links.names
    ]
    if missing_links:
        raise ValueError(
            "The following target links are absent from the robot model: "
            + ", ".join(missing_links)
        )

    target_link_indices_np = np.asarray(
        [robot.links.names.index(name) for name in target_link_names],
        dtype=np.int32,
    )
    target_link_indices_jax = jnp.asarray(target_link_indices_np)

    @jdc.jit
    def _solve_three_arm_ik_jax(
        prev_q: jax.Array,
        target_positions: jax.Array,
        target_wxyzs: jax.Array,
        dt: jax.Array,
        joint_velocity_limits: jax.Array,
        position_weights: jax.Array,
        orientation_weights: jax.Array,
        active_mask: jax.Array,
        dq_weight: jax.Array,
    ) -> jax.Array:
        joint_var = robot.joint_var_cls(0)
        variables = [joint_var]
        costs = []

        # This loop has a fixed length of three and is unrolled during tracing.
        for target_idx in range(NUM_TARGETS):
            T_world_target = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyzs[target_idx]),
                target_positions[target_idx],
            )

            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    T_world_target,
                    target_link_indices_jax[target_idx],
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

        costs.append(pk.costs.limit_constraint(robot, joint_var))

        max_dq = jnp.maximum(joint_velocity_limits * dt, EPS)
        smoothness_scales = dq_weight / max_dq

        costs.append(
            previous_configuration_residual_scaled(
                joint_var=joint_var,
                prev_q=prev_q,
                smoothness_scales=smoothness_scales,
            )
        )

        problem = jaxls.LeastSquaresProblem(
            costs=costs,
            variables=variables,
        )

        solution = (
            problem.analyze().solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(
                    lambda_initial=1.0,
                ),
            )
        )

        return solution[joint_var]

    def solve(
        left_target_position: np.ndarray,
        right_target_position: np.ndarray,
        middle_target_position: np.ndarray,
        left_target_wxyz: np.ndarray,
        right_target_wxyz: np.ndarray,
        middle_target_wxyz: np.ndarray,
        prev_q: np.ndarray,
        dt: float,
        joint_velocity_limits: np.ndarray,
        position_weights: np.ndarray | tuple[float, float, float] = (
            50.0,
            50.0,
            50.0,
        ),
        orientation_weights: np.ndarray | tuple[float, float, float] = (
            2.0,
            2.0,
            2.0,
        ),
        active_mask: np.ndarray | tuple[float, float, float] = (
            1.0,
            1.0,
            1.0,
        ),
        dq_weight: float = 0.5,
        block_until_ready: bool = True,
    ) -> np.ndarray:
        """Solve one coupled three-arm IK step.

        `active_mask` should contain 1.0 for controlled arms and 0.0 for
        inactive arms. An inactive arm has no end-effector pose cost, but the
        previous-configuration residual still discourages it from moving.
        """
        left_target_position_arr = _validate_vector(
            "left_target_position",
            left_target_position,
            (3,),
        )
        right_target_position_arr = _validate_vector(
            "right_target_position",
            right_target_position,
            (3,),
        )
        middle_target_position_arr = _validate_vector(
            "middle_target_position",
            middle_target_position,
            (3,),
        )

        left_target_wxyz_arr = _validate_vector(
            "left_target_wxyz",
            left_target_wxyz,
            (4,),
        )
        right_target_wxyz_arr = _validate_vector(
            "right_target_wxyz",
            right_target_wxyz,
            (4,),
        )
        middle_target_wxyz_arr = _validate_vector(
            "middle_target_wxyz",
            middle_target_wxyz,
            (4,),
        )

        num_joints = robot.joints.num_actuated_joints
        prev_q_arr = _validate_vector("prev_q", prev_q, (num_joints,))
        velocity_limits_arr = _validate_vector(
            "joint_velocity_limits",
            joint_velocity_limits,
            (num_joints,),
        )
        position_weights_arr = _validate_vector(
            "position_weights",
            position_weights,
            (NUM_TARGETS,),
        )
        orientation_weights_arr = _validate_vector(
            "orientation_weights",
            orientation_weights,
            (NUM_TARGETS,),
        )
        active_mask_arr = _validate_vector(
            "active_mask",
            active_mask,
            (NUM_TARGETS,),
        )

        if dt <= 0.0:
            raise ValueError(f"dt must be positive; got {dt}.")
        if dq_weight < 0.0:
            raise ValueError(
                f"dq_weight must be nonnegative; got {dq_weight}."
            )
        if np.any(velocity_limits_arr <= 0.0):
            raise ValueError("All joint velocity limits must be positive.")
        if np.any(position_weights_arr < 0.0):
            raise ValueError("Position weights must be nonnegative.")
        if np.any(orientation_weights_arr < 0.0):
            raise ValueError("Orientation weights must be nonnegative.")
        if np.any((active_mask_arr < 0.0) | (active_mask_arr > 1.0)):
            raise ValueError("active_mask values must lie in [0, 1].")

        target_positions = jnp.asarray(
            np.stack(
                [
                    left_target_position_arr,
                    right_target_position_arr,
                    middle_target_position_arr,
                ],
                axis=0,
            )
        )
        target_wxyzs = jnp.asarray(
            np.stack(
                [
                    left_target_wxyz_arr,
                    right_target_wxyz_arr,
                    middle_target_wxyz_arr,
                ],
                axis=0,
            )
        )

        q_next = _solve_three_arm_ik_jax(
            prev_q=jnp.asarray(prev_q_arr),
            target_positions=target_positions,
            target_wxyzs=target_wxyzs,
            dt=jnp.asarray(dt, dtype=jnp.float32),
            joint_velocity_limits=jnp.asarray(velocity_limits_arr),
            position_weights=jnp.asarray(position_weights_arr),
            orientation_weights=jnp.asarray(orientation_weights_arr),
            active_mask=jnp.asarray(active_mask_arr),
            dq_weight=jnp.asarray(dq_weight, dtype=jnp.float32),
        )

        if block_until_ready:
            q_next = q_next.block_until_ready()

        q_next_np = np.asarray(q_next, dtype=np.float32)

        # Hard safety clamp. The optimizer is softly regularized by the same
        # limits, while this guarantees the returned step cannot exceed them.
        max_dq = velocity_limits_arr * np.float32(dt)
        dq = np.clip(q_next_np - prev_q_arr, -max_dq, max_dq)
        return prev_q_arr + dq

    def warmup(
        prev_q: np.ndarray,
        joint_velocity_limits: np.ndarray,
        dt: float,
        position_weights: np.ndarray | tuple[float, float, float] = (
            50.0,
            50.0,
            50.0,
        ),
        orientation_weights: np.ndarray | tuple[float, float, float] = (
            2.0,
            2.0,
            2.0,
        ),
        active_mask: np.ndarray | tuple[float, float, float] = (
            1.0,
            1.0,
            1.0,
        ),
        dq_weight: float = 0.5,
    ) -> None:
        """Compile the solver using FK targets from the supplied configuration."""
        num_joints = robot.joints.num_actuated_joints
        prev_q_arr = _validate_vector("prev_q", prev_q, (num_joints,))
        fk = robot.forward_kinematics(prev_q_arr)

        positions: list[np.ndarray] = []
        wxyzs: list[np.ndarray] = []

        for link_index in target_link_indices_np:
            T_world_link = jaxlie.SE3(fk[link_index])
            positions.append(
                np.asarray(
                    T_world_link.translation(),
                    dtype=np.float32,
                )
            )
            wxyzs.append(
                np.asarray(
                    T_world_link.rotation().wxyz,
                    dtype=np.float32,
                )
            )

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
            position_weights=position_weights,
            orientation_weights=orientation_weights,
            active_mask=active_mask,
            dq_weight=dq_weight,
            block_until_ready=True,
        )

    return solve, warmup, target_link_indices_np


def solve_ik_three_arm_once(
    robot: pk.Robot,
    left_target_link_name: str,
    right_target_link_name: str,
    middle_target_link_name: str,
    left_target_position: np.ndarray,
    right_target_position: np.ndarray,
    middle_target_position: np.ndarray,
    left_target_wxyz: np.ndarray,
    right_target_wxyz: np.ndarray,
    middle_target_wxyz: np.ndarray,
    prev_q: np.ndarray,
    dt: float,
    joint_velocity_limits: np.ndarray,
    position_weights: np.ndarray | tuple[float, float, float] = (
        50.0,
        50.0,
        50.0,
    ),
    orientation_weights: np.ndarray | tuple[float, float, float] = (
        2.0,
        2.0,
        2.0,
    ),
    active_mask: np.ndarray | tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    ),
    dq_weight: float = 0.5,
) -> np.ndarray:
    """Convenience wrapper for one-off use.

    Repeated control loops should create the solver once with
    `make_three_arm_ik_solver()` and reuse it.
    """
    solve, _, _ = make_three_arm_ik_solver(
        robot=robot,
        left_target_link_name=left_target_link_name,
        right_target_link_name=right_target_link_name,
        middle_target_link_name=middle_target_link_name,
    )

    return solve(
        left_target_position=left_target_position,
        right_target_position=right_target_position,
        middle_target_position=middle_target_position,
        left_target_wxyz=left_target_wxyz,
        right_target_wxyz=right_target_wxyz,
        middle_target_wxyz=middle_target_wxyz,
        prev_q=prev_q,
        dt=dt,
        joint_velocity_limits=joint_velocity_limits,
        position_weights=position_weights,
        orientation_weights=orientation_weights,
        active_mask=active_mask,
        dq_weight=dq_weight,
        block_until_ready=True,
    )
