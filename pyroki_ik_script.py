"""Three-arm PyRoki IK playground with rich collision / manipulability tooling.

Run it, open the printed viser URL, and drag the three gizmos around:

    python pyroki_ik_script.py

The whole thing is one differentiable least-squares problem solved every tick.
Every weight below is a *traced* argument, so moving a slider never triggers a
JAX recompile -- only the "solver iterations" dropdown does (it is static).

Cost terms
----------
  pose              per-arm position + orientation tracking (analytic jacobian)
  manipulability    per-arm, normalized so the weight slider is interpretable
  joint centering   prefer joints near the middle of their range
  limit barrier     one-sided push away from the last few % of each joint range
  nominal pose      weak bias toward the URDF home configuration
  gripper lock      pins the 4 finger joints to the GUI gripper command
  self collision    capsule-vs-capsule over auto-pruned link pairs
  world collision   vs. draggable spheres and an optional ground half-space
  smoothness        penalizes deviation from the previous solution, scaled by
                    the per-tick velocity budget so the weight is unit-free
  limit constraint  hard-ish joint limits via augmented Lagrangian

Why the naive setup misbehaves (and what this script does about it)
------------------------------------------------------------------
1. `RobotCollision.from_urdf` fits ONE capsule per link around the whole
   collision mesh. On this robot that makes 11 link pairs overlap in *every*
   reachable configuration (e.g. `left_base_link` vs `left_upper_arm_link`,
   `middle_camera_body` vs `middle_camera_cover`). Any nonzero self-collision
   weight then fights a battle it can never win and drags the arms off target.
   `auto_ignore_pairs()` samples the configuration space at startup and prunes
   pairs that are in collision essentially always -- those are modeling
   artifacts, not real constraints.
2. PyRoki's manipulability residual is `1 / (yoshikawa + 1e-6)`. On these arms
   yoshikawa sits around 0.02-0.06, so the raw residual is 20-1000 and its
   square is up to ~1e6 -- a weight of 0.05 already swamps the pose cost. Here
   the residual is `weight * m_ref / m`, so it is ~`weight` at a good
   configuration and only grows near singularities.

Visualization
-------------
  * collision capsules, per-link colored by clearance (green -> red)
  * line segments between the closest colliding link pairs, with a live ranking
  * manipulability ellipsoids from the SVD of the translational jacobian,
    colored by the Yoshikawa index
  * target vs. actual end-effector frames and an error segment between them
  * end-effector trails
  * live plots of tracking error, manipulability and collision clearance
  * per-joint sliders showing where each joint sits inside its limits
  * per-term cost breakdown so you can see which cost is actually dominating
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import trimesh
import viser
import yourdfpy
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from viser.extras import ViserUrdf
from yourdfpy import URDF

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

URDF_PATH = Path(__file__).parent / "giava.urdf"

EPS = 1e-6
NUM_OBSTACLES = 3
TRAIL_LENGTH = 200
PLOT_LENGTH = 240
NUM_CLOSEST_PAIRS = 5

# How many recent ticks the mean/max solve-time readouts average over.
TIMING_WINDOW = 30

# Yoshikawa index of a comfortably-posed arm. Used to normalize the
# manipulability residual so its weight slider spans a useful range.
MANIP_REFERENCE = 0.05

# Self-collision pairs colliding in more than this fraction of random
# configurations are treated as capsule-fitting artifacts and pruned.
AUTO_IGNORE_FRACTION = 0.9
AUTO_IGNORE_SAMPLES = 256

# PyRoki raises if any actuated joint lacks `<limit velocity="...">`. Rather
# than require the URDF to carry one, patch a default in memory.
DEFAULT_VELOCITY_LIMIT = np.pi


class ArmSpec(NamedTuple):
    name: str
    ee_link: str
    joint_prefix: str
    color: tuple[int, int, int]


ARMS = (
    ArmSpec("left", "left_gripper_base", "left", (80, 170, 255)),
    ArmSpec("right", "right_gripper_base", "right", (255, 140, 90)),
    ArmSpec("middle", "middle_camera_cover", "middle", (150, 255, 140)),
)
NUM_ARMS = len(ARMS)


# --------------------------------------------------------------------------- #
# Custom residuals
#
# Each is wrapped by `jaxls.Cost.factory`, which strips the leading
# `vals` argument -- the remaining arguments become the factory's arguments.
# --------------------------------------------------------------------------- #


@jaxls.Cost.factory
def joint_centering_residual(vals, joint_var, center, half_range, weights):
    """Prefer joints near the middle of their travel.

    Normalizing by the half-range makes one weight meaningful across joints
    with wildly different spans (0.041 m fingers vs. 2*pi continuous joints):
    the residual is +-`weights` exactly at the limits.
    """
    return (weights * (vals[joint_var] - center) / half_range).flatten()


@jaxls.Cost.factory
def joint_limit_barrier_residual(
    vals, joint_var, center, half_range, activation, weights
):
    """One-sided push that only turns on inside the last of each joint's range.

    Complementary to centering: centering pulls everywhere (and therefore
    always fights the pose cost a little), while this is exactly zero until a
    joint gets within `1 - activation` of a limit.
    """
    normalized = jnp.abs(vals[joint_var] - center) / half_range
    return (weights * jnp.maximum(0.0, normalized - activation)).flatten()


@jaxls.Cost.factory
def masked_rest_residual(vals, joint_var, rest_pose, weights):
    """Per-joint weighted pull toward `rest_pose`.

    Used twice: a weak whole-body bias toward the home pose, and a stiff pin
    holding the gripper fingers at the GUI command (they are actuated joints,
    so otherwise the solver would happily use them as free variables).
    """
    return (weights * (vals[joint_var] - rest_pose)).flatten()


@jaxls.Cost.factory
def smoothness_to_previous_residual(vals, joint_var, prev_q, scales):
    """Penalize motion away from the previous solution.

    `scales` is `weight / max_dq`, so a residual of 1 means "a full tick's
    worth of joint velocity" regardless of dt or the velocity limit.
    """
    return (scales * (vals[joint_var] - prev_q)).flatten()


def _yoshikawa(cfg: jax.Array, robot: pk.Robot, link_index: jax.Array) -> jax.Array:
    """sqrt(det(J J^T)) for the translational jacobian of one link."""
    jacobian = jax.jacfwd(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
    )(cfg)[link_index]
    return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(jacobian @ jacobian.T)))


@jaxls.Cost.factory
def normalized_manipulability_residual(
    vals, robot, joint_var, link_index, reference, weight
):
    """`weight * m_ref / m`: ~`weight` when well-conditioned, blows up at
    singularities. See the module docstring for why the raw PyRoki residual is
    hard to weight on this robot."""
    manip = _yoshikawa(vals[joint_var], robot, link_index)
    return (weight * reference / (manip + EPS)).flatten()


# --------------------------------------------------------------------------- #
# URDF loading
# --------------------------------------------------------------------------- #


def load_urdf(path: Path) -> URDF:
    """Load the URDF, filling in velocity limits PyRoki insists on.

    The velocity limit only feeds PyRoki's `limit_velocity_*` costs, which this
    script does not use -- it clamps dq directly against the GUI slider -- so a
    default is harmless and beats refusing to start.
    """
    urdf = URDF.load(str(path))
    patched = []
    for name, joint in urdf.joint_map.items():
        if joint.type == "fixed":
            continue
        if joint.limit is None:
            joint.limit = yourdfpy.Limit(
                lower=-DEFAULT_VELOCITY_LIMIT,
                upper=DEFAULT_VELOCITY_LIMIT,
                velocity=DEFAULT_VELOCITY_LIMIT,
            )
            patched.append(name)
        elif joint.limit.velocity is None:
            joint.limit.velocity = DEFAULT_VELOCITY_LIMIT
            patched.append(name)
    if patched:
        print(
            f"  no <limit velocity=...> on {len(patched)} joint(s) "
            f"({', '.join(patched)}); defaulting to {DEFAULT_VELOCITY_LIMIT:.3f}"
        )
    return urdf


# --------------------------------------------------------------------------- #
# Self-collision pair pruning
# --------------------------------------------------------------------------- #


def auto_ignore_pairs(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    num_samples: int = AUTO_IGNORE_SAMPLES,
    fraction: float = AUTO_IGNORE_FRACTION,
    seed: int = 0,
) -> tuple[tuple[str, str], ...]:
    """Find link pairs that are 'in collision' almost everywhere.

    One capsule per link is a coarse fit, so some pairs overlap in every
    configuration the robot can reach. Those are modeling artifacts: keeping
    them makes the self-collision residual a large constant offset that pulls
    the solution away from the targets no matter what. We detect them by
    sampling the joint space and pruning pairs that collide almost always.
    """
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    rng = np.random.default_rng(seed)
    cfgs = rng.uniform(lower, upper, size=(num_samples, lower.shape[0]))

    batched = jax.jit(
        jax.vmap(lambda c: robot_coll.compute_self_collision_distance(robot, c))
    )
    distances = np.asarray(batched(jnp.asarray(cfgs, dtype=jnp.float32)))

    collide_fraction = (distances < 0.0).mean(axis=0)
    ignore = []
    for k in np.where(collide_fraction >= fraction)[0]:
        ignore.append(
            (
                robot_coll.link_names[robot_coll.active_idx_i[k]],
                robot_coll.link_names[robot_coll.active_idx_j[k]],
            )
        )
    return tuple(ignore)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@jdc.pytree_dataclass
class SolveWeights:
    """Every tunable knob, all traced -- moving a slider never recompiles."""

    position: jax.Array  # (NUM_ARMS,)
    orientation: jax.Array  # (NUM_ARMS,)
    manipulability: jax.Array  # (NUM_ARMS,)
    centering: jax.Array  # ()
    barrier: jax.Array  # ()
    barrier_activation: jax.Array  # ()
    nominal: jax.Array  # ()
    gripper_hold: jax.Array  # ()
    smoothness: jax.Array  # ()
    self_collision: jax.Array  # ()
    world_collision: jax.Array  # ()
    ground_collision: jax.Array  # ()
    collision_margin: jax.Array  # ()


def make_solver(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    ee_link_indices: np.ndarray,
    centering_mask: np.ndarray,
    finger_mask: np.ndarray,
) -> Callable[..., tuple[np.ndarray, float, float]]:
    """Build the jitted IK step.

    `centering_mask` zeroes centering/barrier on the finger joints (whose
    "center" is a half-open gripper, which is not a preference we want).
    `finger_mask` selects those same joints for the gripper pin.
    """
    ee_indices = jnp.asarray(ee_link_indices, dtype=jnp.int32)
    centering_mask_j = jnp.asarray(centering_mask, dtype=jnp.float32)
    finger_mask_j = jnp.asarray(finger_mask, dtype=jnp.float32)

    lower = jnp.asarray(robot.joints.lower_limits, dtype=jnp.float32)
    upper = jnp.asarray(robot.joints.upper_limits, dtype=jnp.float32)
    center = 0.5 * (lower + upper)
    half_range = jnp.maximum(0.5 * (upper - lower), EPS)

    @jdc.jit
    def _solve(
        prev_q: jax.Array,
        target_positions: jax.Array,
        target_wxyzs: jax.Array,
        nominal_q: jax.Array,
        gripper_q: jax.Array,
        max_dq: jax.Array,
        weights: SolveWeights,
        obstacles: Sphere,
        ground: HalfSpace,
        max_iterations: jdc.Static[int],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        joint_var = robot.joint_var_cls(0)
        costs = []

        for arm in range(NUM_ARMS):
            target_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyzs[arm]),
                target_positions[arm],
            )
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    target_pose,
                    ee_indices[arm],
                    pos_weight=weights.position[arm],
                    ori_weight=weights.orientation[arm],
                )
            )
            costs.append(
                normalized_manipulability_residual(
                    robot=robot,
                    joint_var=joint_var,
                    link_index=ee_indices[arm],
                    reference=jnp.asarray(MANIP_REFERENCE, dtype=jnp.float32),
                    weight=weights.manipulability[arm],
                )
            )

        costs.append(
            joint_centering_residual(
                joint_var=joint_var,
                center=center,
                half_range=half_range,
                weights=weights.centering * centering_mask_j,
            )
        )
        costs.append(
            joint_limit_barrier_residual(
                joint_var=joint_var,
                center=center,
                half_range=half_range,
                activation=weights.barrier_activation,
                weights=weights.barrier * centering_mask_j,
            )
        )
        costs.append(
            masked_rest_residual(
                joint_var=joint_var,
                rest_pose=nominal_q,
                weights=weights.nominal * centering_mask_j,
            )
        )
        costs.append(
            masked_rest_residual(
                joint_var=joint_var,
                rest_pose=gripper_q,
                weights=weights.gripper_hold * finger_mask_j,
            )
        )

        costs.append(pk.costs.limit_constraint(robot, joint_var))

        costs.append(
            pk.costs.self_collision_cost(
                robot=robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                margin=weights.collision_margin,
                weight=weights.self_collision,
            )
        )
        costs.append(
            pk.costs.world_collision_cost(
                robot=robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                world_geom=obstacles,
                margin=weights.collision_margin,
                weight=weights.world_collision,
            )
        )
        costs.append(
            pk.costs.world_collision_cost(
                robot=robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                world_geom=ground,
                margin=weights.collision_margin,
                weight=weights.ground_collision,
            )
        )

        costs.append(
            smoothness_to_previous_residual(
                joint_var=joint_var,
                prev_q=prev_q,
                scales=weights.smoothness / jnp.maximum(max_dq, EPS),
            )
        )

        problem = jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        solution, summary = problem.analyze().solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=max_iterations),
            return_summary=True,
        )
        final_cost = summary.cost_history[summary.iterations]
        return solution[joint_var], final_cost, summary.iterations

    def solve(
        *,
        prev_q: np.ndarray,
        target_positions: np.ndarray,
        target_wxyzs: np.ndarray,
        nominal_q: np.ndarray,
        gripper_q: np.ndarray,
        velocity_limits: np.ndarray,
        dt: float,
        weights: SolveWeights,
        obstacles: Sphere,
        ground: HalfSpace,
        max_iterations: int,
    ) -> tuple[np.ndarray, float, float]:
        max_dq = np.maximum(velocity_limits * np.float32(dt), EPS).astype(np.float32)
        q_next, cost, iterations = _solve(
            prev_q=jnp.asarray(prev_q, dtype=jnp.float32),
            target_positions=jnp.asarray(target_positions, dtype=jnp.float32),
            target_wxyzs=jnp.asarray(target_wxyzs, dtype=jnp.float32),
            nominal_q=jnp.asarray(nominal_q, dtype=jnp.float32),
            gripper_q=jnp.asarray(gripper_q, dtype=jnp.float32),
            max_dq=jnp.asarray(max_dq),
            weights=weights,
            obstacles=obstacles,
            ground=ground,
            max_iterations=max_iterations,
        )
        q_next = np.asarray(jax.block_until_ready(q_next), dtype=np.float32)

        # The smoothness cost is soft; this makes the velocity limit hard.
        dq = np.clip(q_next - prev_q, -max_dq, max_dq)
        return prev_q + dq, float(cost), float(iterations)

    return solve


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def make_diagnostics(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    ee_link_indices: np.ndarray,
) -> Callable[..., dict[str, np.ndarray]]:
    """One jitted pass computing everything the GUI and the scene want to show."""
    ee_indices = jnp.asarray(ee_link_indices, dtype=jnp.int32)
    num_links = robot_coll.num_links
    idx_i = jnp.asarray(robot_coll.active_idx_i, dtype=jnp.int32)
    idx_j = jnp.asarray(robot_coll.active_idx_j, dtype=jnp.int32)

    @jax.jit
    def _diagnose(cfg: jax.Array, obstacles: Sphere, ground: HalfSpace):
        fk = robot.forward_kinematics(cfg)
        poses = jaxlie.SE3(fk)

        ee_pose = jaxlie.SE3(fk[ee_indices])
        ee_positions = ee_pose.translation()
        ee_wxyzs = ee_pose.rotation().wxyz

        # Translational jacobian per end effector -> SVD gives both the
        # manipulability ellipsoid axes and the Yoshikawa index / condition.
        full_jac = jax.jacfwd(
            lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
        )(cfg)
        jacobians = full_jac[ee_indices]  # (NUM_ARMS, 3, num_joints)
        u_mats, sing_vals, _ = jnp.linalg.svd(jacobians, full_matrices=False)

        self_distances = robot_coll.compute_self_collision_distance(robot, cfg)

        # Per-link clearance, for coloring the capsules.
        big = jnp.full((num_links,), 1e3)
        link_self_min = big.at[idx_i].min(self_distances)
        link_self_min = link_self_min.at[idx_j].min(self_distances)

        obstacle_distances = robot_coll.compute_world_collision_distance(
            robot, cfg, obstacles
        )  # (num_links, num_obstacles)
        ground_distances = robot_coll.compute_world_collision_distance(
            robot, cfg, ground
        )  # (num_links, 1)
        link_world_min = jnp.minimum(
            obstacle_distances.min(axis=-1), ground_distances.min(axis=-1)
        )

        coll_world = robot_coll.at_config(robot, cfg)
        capsule_centers = coll_world.pose.translation()

        return {
            "ee_positions": ee_positions,
            "ee_wxyzs": ee_wxyzs,
            "link_positions": poses.translation(),
            "link_wxyzs": poses.rotation().wxyz,
            "ellipsoid_axes": u_mats,
            "singular_values": sing_vals,
            "manipulability": jnp.prod(sing_vals, axis=-1),
            "condition": sing_vals[..., 0] / jnp.maximum(sing_vals[..., -1], EPS),
            "self_distances": self_distances,
            "link_self_min": link_self_min,
            "link_world_min": link_world_min,
            "link_min": jnp.minimum(link_self_min, link_world_min),
            "capsule_centers": capsule_centers,
        }

    def diagnose(cfg, obstacles, ground):
        out = _diagnose(jnp.asarray(cfg, dtype=jnp.float32), obstacles, ground)
        return {k: np.asarray(v) for k, v in out.items()}

    return diagnose


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def clearance_color(distance: float, margin: float) -> tuple[int, int, int]:
    """Green when clear, yellow inside the margin, red when penetrating."""
    if distance < 0.0:
        return (255, 40, 40)
    t = float(np.clip(distance / max(margin, 1e-3), 0.0, 1.0))
    if t < 0.5:
        s = t / 0.5  # red -> yellow
        return (255, int(60 + 195 * s), 40)
    s = (t - 0.5) / 0.5  # yellow -> green
    return (int(255 * (1.0 - s) + 90 * s), 255, int(40 + 100 * s))


def manipulability_color(manip: float) -> tuple[int, int, int]:
    """Blue-ish when well conditioned, red as the arm approaches a singularity."""
    t = float(np.clip(manip / (2.0 * MANIP_REFERENCE), 0.0, 1.0))
    return (int(255 * (1.0 - t) + 90 * t), int(80 + 120 * t), int(80 + 175 * t))


def pose_from_fk(fk: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    pose = jaxlie.SE3(jnp.asarray(fk[index]))
    return (
        np.asarray(pose.translation(), dtype=np.float32),
        np.asarray(pose.rotation().wxyz, dtype=np.float32),
    )


def quaternion_angle(wxyz_a: np.ndarray, wxyz_b: np.ndarray) -> float:
    """Geodesic angle between two quaternions, in radians."""
    dot = float(np.clip(abs(float(np.dot(wxyz_a, wxyz_b))), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


class RingBuffer:
    """Fixed-length history for the live plots."""

    def __init__(self, length: int, width: int = 1):
        self._data = np.zeros((length, width), dtype=np.float32)
        self._count = 0

    def push(self, value) -> None:
        self._data = np.roll(self._data, -1, axis=0)
        self._data[-1] = value
        self._count = min(self._count + 1, self._data.shape[0])

    @property
    def values(self) -> np.ndarray:
        return self._data[-self._count :] if self._count else self._data[:0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    print(f"Loading {URDF_PATH} ...")
    urdf = load_urdf(URDF_PATH)
    robot = pk.Robot.from_urdf(urdf)

    joint_names = list(robot.joints.actuated_names)
    num_joints = robot.joints.num_actuated_joints
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float32)
    center = 0.5 * (lower + upper)
    half_range = np.maximum(0.5 * (upper - lower), EPS)

    finger_mask = np.asarray(
        [1.0 if "finger" in n else 0.0 for n in joint_names], dtype=np.float32
    )
    centering_mask = 1.0 - finger_mask
    finger_indices = np.where(finger_mask > 0.0)[0]
    finger_upper = float(upper[finger_indices[0]]) if finger_indices.size else 0.041

    nominal_q = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    cfg = nominal_q.copy()

    print("Building collision model ...")
    coarse_coll = RobotCollision.from_urdf(urdf)
    print(f"Pruning always-colliding pairs ({AUTO_IGNORE_SAMPLES} samples) ...")
    ignore_pairs = auto_ignore_pairs(robot, coarse_coll)
    for a, b in ignore_pairs:
        print(f"  ignoring {a} <-> {b}")
    robot_coll = RobotCollision.from_urdf(urdf, user_ignore_pairs=ignore_pairs)

    ee_link_indices = np.asarray(
        [robot.links.names.index(arm.ee_link) for arm in ARMS], dtype=np.int32
    )

    solve = make_solver(robot, robot_coll, ee_link_indices, centering_mask, finger_mask)
    diagnose = make_diagnostics(robot, robot_coll, ee_link_indices)

    # ----------------------------------------------------------------- scene #
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(cfg)

    fk = np.asarray(robot.forward_kinematics(cfg))

    target_controls = []
    ee_frames = []
    error_segments = []
    trail_clouds = []
    trails = [np.zeros((0, 3), dtype=np.float32) for _ in ARMS]
    for arm_idx, arm in enumerate(ARMS):
        position, wxyz = pose_from_fk(fk, int(ee_link_indices[arm_idx]))
        control = server.scene.add_transform_controls(
            f"/targets/{arm.name}", scale=0.15, position=position, wxyz=wxyz
        )
        server.scene.add_icosphere(
            f"/targets/{arm.name}/marker", radius=0.012, color=arm.color, opacity=0.6
        )
        target_controls.append(control)
        ee_frames.append(
            server.scene.add_frame(
                f"/actual/{arm.name}",
                axes_length=0.06,
                axes_radius=0.003,
                position=position,
                wxyz=wxyz,
            )
        )
        error_segments.append(
            server.scene.add_line_segments(
                f"/error/{arm.name}",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=np.array([[arm.color, arm.color]], dtype=np.uint8),
                line_width=3.0,
            )
        )
        trail_clouds.append(
            server.scene.add_point_cloud(
                f"/trails/{arm.name}",
                points=np.zeros((1, 3), dtype=np.float32),
                colors=np.array(arm.color, dtype=np.uint8),
                point_size=0.004,
                point_shape="circle",
                visible=False,
            )
        )

    # Collision capsules: built once in each link's local frame, then parented
    # to a frame we drive from FK. Per tick we only touch the color.
    local_meshes = robot_coll.get_link_collision_meshes()
    capsule_frames: dict[int, viser.FrameHandle] = {}
    capsule_meshes: dict[int, viser.MeshHandle] = {}
    capsule_colors: dict[int, tuple[int, int, int]] = {}
    for link_idx, link_name in enumerate(robot_coll.link_names):
        mesh = local_meshes.get(link_name)
        if mesh is None or len(mesh.vertices) == 0:
            continue
        capsule_frames[link_idx] = server.scene.add_frame(
            f"/collision/{link_name}", show_axes=False, visible=False
        )
        capsule_meshes[link_idx] = server.scene.add_mesh_simple(
            f"/collision/{link_name}/capsule",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=(90, 220, 120),
            wireframe=True,
            opacity=0.45,
            side="double",
        )
        capsule_colors[link_idx] = (90, 220, 120)

    closest_pair_segments = server.scene.add_line_segments(
        "/collision_pairs",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.array([[[255, 60, 60], [255, 60, 60]]], dtype=np.uint8),
        line_width=4.0,
        visible=False,
    )

    base_ellipsoid = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    ellipsoid_vertices = np.asarray(base_ellipsoid.vertices, dtype=np.float32)
    ellipsoid_faces = np.asarray(base_ellipsoid.faces, dtype=np.uint32)
    ellipsoid_meshes = [
        server.scene.add_mesh_simple(
            f"/manipulability/{arm.name}",
            vertices=ellipsoid_vertices,
            faces=ellipsoid_faces,
            color=arm.color,
            wireframe=True,
            opacity=0.5,
            side="double",
        )
        for arm in ARMS
    ]

    obstacle_controls = []
    obstacle_meshes = []
    for i in range(NUM_OBSTACLES):
        obstacle_controls.append(
            server.scene.add_transform_controls(
                f"/obstacles/{i}",
                scale=0.12,
                disable_rotations=True,
                position=(0.35, -0.25 + 0.25 * i, 0.4),
                visible=False,
            )
        )
        obstacle_meshes.append(
            server.scene.add_icosphere(
                f"/obstacles/{i}/mesh",
                radius=0.08,
                color=(230, 120, 230),
                opacity=0.55,
                visible=False,
            )
        )

    ground_plane = server.scene.add_box(
        "/ground_collision",
        dimensions=(3.0, 3.0, 0.005),
        color=(120, 120, 160),
        opacity=0.35,
        visible=False,
    )

    # ------------------------------------------------------------------- gui #
    tabs = server.gui.add_tab_group()

    with tabs.add_tab("Run"):
        running = server.gui.add_checkbox("Solve", True)
        rate_slider = server.gui.add_slider("Target rate (Hz)", 5, 120, 1, 30)
        velocity_slider = server.gui.add_slider(
            "Joint velocity limit (rad/s)", 0.1, 6.0, 0.1, 2.0
        )
        iteration_dropdown = server.gui.add_dropdown(
            "Solver iterations (recompiles)",
            ("4", "8", "16", "32", "64"),
            initial_value="16",
        )
        gripper_slider = server.gui.add_slider(
            "Gripper opening (m)", 0.0, finger_upper, 0.001, finger_upper
        )

        server.gui.add_markdown("---")
        snap_button = server.gui.add_button("Snap targets to current EE")
        home_button = server.gui.add_button("Go home + snap targets")
        randomize_button = server.gui.add_button("Randomize targets (from FK)")
        nudge_button = server.gui.add_button("Perturb configuration")

        server.gui.add_markdown("**Arms**")
        arm_enables = [server.gui.add_checkbox(f"Track {arm.name}", True) for arm in ARMS]

    with tabs.add_tab("Weights"):
        position_weights = []
        orientation_weights = []
        manipulability_weights = []
        for arm in ARMS:
            with server.gui.add_folder(arm.name.capitalize()):
                position_weights.append(
                    server.gui.add_slider("position", 0.0, 200.0, 1.0, 50.0)
                )
                orientation_weights.append(
                    server.gui.add_slider("orientation", 0.0, 50.0, 0.1, 5.0)
                )
                manipulability_weights.append(
                    server.gui.add_slider("manipulability", 0.0, 20.0, 0.05, 0.0)
                )

        with server.gui.add_folder("Regularization"):
            centering_weight = server.gui.add_slider(
                "joint centering", 0.0, 20.0, 0.05, 1.0
            )
            barrier_weight = server.gui.add_slider("limit barrier", 0.0, 200.0, 0.5, 20.0)
            barrier_activation = server.gui.add_slider(
                "barrier turn-on (frac of range)", 0.5, 1.0, 0.01, 0.9
            )
            nominal_weight = server.gui.add_slider("nominal pose", 0.0, 20.0, 0.05, 0.0)
            smoothness_weight = server.gui.add_slider("smoothness", 0.0, 10.0, 0.01, 0.5)
            gripper_hold_weight = server.gui.add_slider(
                "gripper hold", 0.0, 2000.0, 10.0, 500.0
            )

        with server.gui.add_folder("Collision"):
            self_collision_weight = server.gui.add_slider(
                "self collision", 0.0, 200.0, 1.0, 20.0
            )
            world_collision_weight = server.gui.add_slider(
                "world collision", 0.0, 200.0, 1.0, 20.0
            )
            ground_collision_weight = server.gui.add_slider(
                "ground collision", 0.0, 200.0, 1.0, 0.0
            )
            collision_margin = server.gui.add_slider("margin (m)", 0.0, 0.20, 0.005, 0.03)

        reset_weights_button = server.gui.add_button("Reset weights to defaults")

    with tabs.add_tab("Collision"):
        show_capsules = server.gui.add_checkbox("Show collision capsules", False)
        show_robot_mesh = server.gui.add_checkbox("Show robot meshes", True)
        show_pair_lines = server.gui.add_checkbox("Show closest pairs", True)
        min_self_readout = server.gui.add_number("Min self distance (m)", 0.0, disabled=True)
        min_world_readout = server.gui.add_number(
            "Min world distance (m)", 0.0, disabled=True
        )
        in_collision_readout = server.gui.add_number(
            "Pairs inside margin", 0, disabled=True
        )
        closest_pairs_text = server.gui.add_markdown("")

        with server.gui.add_folder("Obstacles"):
            obstacle_enables = []
            obstacle_radii = []
            for i in range(NUM_OBSTACLES):
                obstacle_enables.append(
                    server.gui.add_checkbox(f"sphere {i} enabled", False)
                )
                obstacle_radii.append(
                    server.gui.add_slider(f"sphere {i} radius", 0.02, 0.30, 0.01, 0.08)
                )
            ground_height = server.gui.add_slider("ground plane z (m)", -0.5, 0.6, 0.01, 0.0)
            show_ground_plane = server.gui.add_checkbox("Show ground plane", False)

        server.gui.add_markdown(
            f"Auto-pruned **{len(ignore_pairs)}** always-colliding link pairs; "
            f"**{len(robot_coll.active_idx_i)}** pairs remain active."
        )

    with tabs.add_tab("Manipulability"):
        show_ellipsoids = server.gui.add_checkbox("Show ellipsoids", True)
        ellipsoid_scale = server.gui.add_slider("Ellipsoid scale", 0.01, 1.0, 0.01, 0.2)
        color_by_manip = server.gui.add_checkbox("Color by manipulability", True)
        manip_readouts = []
        condition_readouts = []
        for arm in ARMS:
            with server.gui.add_folder(arm.name.capitalize()):
                manip_readouts.append(
                    server.gui.add_number("Yoshikawa index", 0.0, disabled=True)
                )
                condition_readouts.append(
                    server.gui.add_number("Condition number", 0.0, disabled=True)
                )
        manip_plot = server.gui.add_uplot(
            data=(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)),
            series=(
                {"label": "t"},
                {"label": "left", "stroke": "rgb(80,170,255)"},
                {"label": "right", "stroke": "rgb(255,140,90)"},
                {"label": "middle", "stroke": "rgb(150,255,140)"},
            ),
            title="Yoshikawa index",
            aspect=2.0,
        )

    with tabs.add_tab("Diagnostics"):
        with server.gui.add_folder("Timing"):
            # "Solve" is one IK step: the whole jitted least-squares problem.
            # "Iterations" is how many Levenberg-Marquardt steps that took,
            # so solve / iterations is the cost of one LM iteration.
            solve_time_readout = server.gui.add_number(
                "Solve time (ms)", 0.0, disabled=True
            )
            solve_avg_readout = server.gui.add_number(
                f"Solve time, mean of {TIMING_WINDOW} (ms)", 0.0, disabled=True
            )
            solve_max_readout = server.gui.add_number(
                f"Solve time, max of {TIMING_WINDOW} (ms)", 0.0, disabled=True
            )
            iterations_readout = server.gui.add_number("Iterations", 0.0, disabled=True)
            per_iteration_readout = server.gui.add_number(
                "ms / solver iteration", 0.0, disabled=True
            )
            diagnostics_time_readout = server.gui.add_number(
                "Diagnostics + draw (ms)", 0.0, disabled=True
            )
            loop_time_readout = server.gui.add_number(
                "Whole tick (ms)", 0.0, disabled=True
            )
            loop_rate_readout = server.gui.add_number("Loop rate (Hz)", 0.0, disabled=True)
            solve_time_plot = server.gui.add_uplot(
                data=(np.zeros(1), np.zeros(1), np.zeros(1)),
                series=(
                    {"label": "t"},
                    {"label": "solve", "stroke": "rgb(120,200,255)"},
                    {"label": "per iter", "stroke": "rgb(255,200,120)"},
                ),
                title="Solve time (ms)",
                aspect=2.0,
            )

        total_cost_readout = server.gui.add_number("Total cost", 0.0, disabled=True)

        with server.gui.add_folder("Tracking error"):
            position_error_readouts = []
            orientation_error_readouts = []
            for arm in ARMS:
                position_error_readouts.append(
                    server.gui.add_number(f"{arm.name} pos (mm)", 0.0, disabled=True)
                )
                orientation_error_readouts.append(
                    server.gui.add_number(f"{arm.name} ori (deg)", 0.0, disabled=True)
                )

        error_plot = server.gui.add_uplot(
            data=(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)),
            series=(
                {"label": "t"},
                {"label": "left", "stroke": "rgb(80,170,255)"},
                {"label": "right", "stroke": "rgb(255,140,90)"},
                {"label": "middle", "stroke": "rgb(150,255,140)"},
            ),
            title="Position error (mm)",
            aspect=2.0,
        )
        clearance_plot = server.gui.add_uplot(
            data=(np.zeros(1), np.zeros(1), np.zeros(1)),
            series=(
                {"label": "t"},
                {"label": "self", "stroke": "rgb(255,90,90)"},
                {"label": "world", "stroke": "rgb(230,120,230)"},
            ),
            title="Min clearance (m)",
            aspect=2.0,
        )

        with server.gui.add_folder("Cost breakdown (weighted, squared)"):
            cost_readouts = {
                key: server.gui.add_number(key, 0.0, disabled=True)
                for key in (
                    "pose",
                    "manipulability",
                    "centering",
                    "barrier",
                    "nominal",
                    "self collision",
                    "world collision",
                    "solver total",
                )
            }

        show_trails = server.gui.add_checkbox("Show EE trails", False)
        with server.gui.add_folder("Joint limits"):
            worst_limit_readout = server.gui.add_number(
                "Closest joint to limit (frac)", 0.0, disabled=True
            )
            joint_sliders = []
            for arm in ARMS:
                with server.gui.add_folder(arm.name.capitalize(), expand_by_default=False):
                    for j, name in enumerate(joint_names):
                        if not name.startswith(arm.joint_prefix):
                            continue
                        joint_sliders.append(
                            (
                                j,
                                server.gui.add_slider(
                                    name[len(arm.joint_prefix) :],
                                    float(lower[j]),
                                    float(upper[j]),
                                    0.001,
                                    float(cfg[j]),
                                    disabled=True,
                                ),
                            )
                        )

    # -------------------------------------------------------------- callbacks #
    # GUI handles are unhashable, so keep (handle, default) pairs in a list.
    weight_defaults = [
        (handle, handle.value)
        for handle in (
            *position_weights,
            *orientation_weights,
            *manipulability_weights,
            centering_weight,
            barrier_weight,
            barrier_activation,
            nominal_weight,
            smoothness_weight,
            gripper_hold_weight,
            self_collision_weight,
            world_collision_weight,
            ground_collision_weight,
            collision_margin,
        )
    ]

    @reset_weights_button.on_click
    def _(_) -> None:
        for handle, value in weight_defaults:
            handle.value = value

    def snap_targets(source_cfg: np.ndarray) -> None:
        source_fk = np.asarray(robot.forward_kinematics(source_cfg))
        for arm_idx in range(NUM_ARMS):
            position, wxyz = pose_from_fk(source_fk, int(ee_link_indices[arm_idx]))
            target_controls[arm_idx].position = position
            target_controls[arm_idx].wxyz = wxyz

    @snap_button.on_click
    def _(_) -> None:
        snap_targets(cfg)

    @home_button.on_click
    def _(_) -> None:
        nonlocal cfg
        cfg = nominal_q.copy()
        snap_targets(cfg)

    rng = np.random.default_rng()

    @randomize_button.on_click
    def _(_) -> None:
        # Sample a random configuration and use its FK as the targets, so the
        # targets are always reachable by at least one configuration.
        snap_targets(rng.uniform(lower, upper).astype(np.float32))

    @nudge_button.on_click
    def _(_) -> None:
        nonlocal cfg
        cfg = np.clip(
            cfg + rng.normal(0.0, 0.15, size=num_joints).astype(np.float32), lower, upper
        )

    # ----------------------------------------------------------------- warmup #
    def current_weights() -> SolveWeights:
        enables = np.array([float(h.value) for h in arm_enables], dtype=np.float32)

        def f32(x):
            return jnp.asarray(x, dtype=jnp.float32)

        return SolveWeights(
            position=f32([h.value for h in position_weights]) * f32(enables),
            orientation=f32([h.value for h in orientation_weights]) * f32(enables),
            manipulability=f32([h.value for h in manipulability_weights]),
            centering=f32(centering_weight.value),
            barrier=f32(barrier_weight.value),
            barrier_activation=f32(barrier_activation.value),
            nominal=f32(nominal_weight.value),
            gripper_hold=f32(gripper_hold_weight.value),
            smoothness=f32(smoothness_weight.value),
            self_collision=f32(self_collision_weight.value),
            world_collision=f32(world_collision_weight.value),
            ground_collision=f32(ground_collision_weight.value),
            collision_margin=f32(collision_margin.value),
        )

    obstacle_geom = Sphere.from_center_and_radius(
        jnp.zeros((NUM_OBSTACLES, 3)), jnp.full((NUM_OBSTACLES,), 0.08)
    )
    ground_geom = HalfSpace.from_point_and_normal(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    )

    print("Compiling solver (the first solve is slow) ...")
    compile_start = time.perf_counter()
    solve(
        prev_q=cfg,
        target_positions=np.stack(
            [np.asarray(h.position, dtype=np.float32) for h in target_controls]
        ),
        target_wxyzs=np.stack(
            [np.asarray(h.wxyz, dtype=np.float32) for h in target_controls]
        ),
        nominal_q=nominal_q,
        gripper_q=np.full(num_joints, gripper_slider.value, dtype=np.float32),
        velocity_limits=np.full(num_joints, 2.0, dtype=np.float32),
        dt=1.0 / 30.0,
        weights=current_weights(),
        obstacles=obstacle_geom,
        ground=ground_geom,
        max_iterations=int(iteration_dropdown.value),
    )
    diagnose(cfg, obstacle_geom, ground_geom)
    print(f"Ready in {time.perf_counter() - compile_start:.1f}s.")

    # -------------------------------------------------------------- main loop #
    time_history = RingBuffer(PLOT_LENGTH)
    manip_history = RingBuffer(PLOT_LENGTH, NUM_ARMS)
    error_history = RingBuffer(PLOT_LENGTH, NUM_ARMS)
    clearance_history = RingBuffer(PLOT_LENGTH, 2)
    solve_time_history = RingBuffer(PLOT_LENGTH, 2)  # (solve ms, ms per iteration)

    start_time = time.perf_counter()
    last_loop = start_time
    slow_counter = 0

    try:
        while True:
            loop_start = time.perf_counter()
            dt = max(1.0 / float(rate_slider.value), 1e-3)

            # ---- gather inputs ----
            target_positions = np.stack(
                [np.asarray(h.position, dtype=np.float32) for h in target_controls]
            )
            target_wxyzs = np.stack(
                [np.asarray(h.wxyz, dtype=np.float32) for h in target_controls]
            )

            obstacle_centers = np.zeros((NUM_OBSTACLES, 3), dtype=np.float32)
            obstacle_radii_arr = np.zeros(NUM_OBSTACLES, dtype=np.float32)
            for i in range(NUM_OBSTACLES):
                if obstacle_enables[i].value:
                    obstacle_centers[i] = obstacle_controls[i].position
                    obstacle_radii_arr[i] = obstacle_radii[i].value
                else:
                    # Radius 0, parked far below the floor: no penalty, and the
                    # shape stays fixed so nothing recompiles.
                    obstacle_centers[i] = (0.0, 0.0, -100.0)
                    obstacle_radii_arr[i] = 0.0
            obstacle_geom = Sphere.from_center_and_radius(
                jnp.asarray(obstacle_centers), jnp.asarray(obstacle_radii_arr)
            )
            ground_geom = HalfSpace.from_point_and_normal(
                jnp.array([0.0, 0.0, float(ground_height.value)]),
                jnp.array([0.0, 0.0, 1.0]),
            )

            gripper_q = np.zeros(num_joints, dtype=np.float32)
            gripper_q[finger_indices] = gripper_slider.value

            weights = current_weights()
            velocity_limits = np.full(
                num_joints, float(velocity_slider.value), dtype=np.float32
            )

            # ---- solve ----
            solve_start = time.perf_counter()
            if running.value:
                cfg, total_cost, iterations = solve(
                    prev_q=cfg,
                    target_positions=target_positions,
                    target_wxyzs=target_wxyzs,
                    nominal_q=nominal_q,
                    gripper_q=gripper_q,
                    velocity_limits=velocity_limits,
                    dt=dt,
                    weights=weights,
                    obstacles=obstacle_geom,
                    ground=ground_geom,
                    max_iterations=int(iteration_dropdown.value),
                )
            else:
                total_cost, iterations = 0.0, 0.0
            solve_ms = (time.perf_counter() - solve_start) * 1e3
            # `iterations` is the LM step count the solver actually used, which
            # can be below the cap when it converges early -- so this is the
            # real per-iteration cost, not solve_ms / max_iterations.
            per_iteration_ms = solve_ms / max(iterations, 1.0)

            draw_start = time.perf_counter()
            info = diagnose(cfg, obstacle_geom, ground_geom)

            # ---- robot, targets, tracking error ----
            urdf_vis.update_cfg(cfg)
            position_errors = np.linalg.norm(
                info["ee_positions"] - target_positions, axis=-1
            )
            for arm_idx in range(NUM_ARMS):
                ee_position = info["ee_positions"][arm_idx]
                ee_wxyz = info["ee_wxyzs"][arm_idx]
                ee_frames[arm_idx].position = ee_position
                ee_frames[arm_idx].wxyz = ee_wxyz
                target_controls[arm_idx].visible = arm_enables[arm_idx].value
                error_segments[arm_idx].points = np.array(
                    [[ee_position, target_positions[arm_idx]]], dtype=np.float32
                )
                position_error_readouts[arm_idx].value = round(
                    float(position_errors[arm_idx] * 1e3), 3
                )
                orientation_error_readouts[arm_idx].value = round(
                    float(np.degrees(quaternion_angle(ee_wxyz, target_wxyzs[arm_idx]))), 3
                )

            # ---- trails ----
            if show_trails.value:
                for arm_idx in range(NUM_ARMS):
                    trails[arm_idx] = np.concatenate(
                        [trails[arm_idx], info["ee_positions"][arm_idx][None]]
                    )[-TRAIL_LENGTH:]
                    trail_clouds[arm_idx].points = trails[arm_idx]
                    trail_clouds[arm_idx].visible = True
            else:
                for arm_idx in range(NUM_ARMS):
                    trail_clouds[arm_idx].visible = False
                    trails[arm_idx] = np.zeros((0, 3), dtype=np.float32)

            # ---- collision visuals ----
            urdf_vis.show_visual = show_robot_mesh.value
            margin = float(collision_margin.value)
            if show_capsules.value:
                link_min = info["link_min"]
                for link_idx, frame in capsule_frames.items():
                    frame.visible = True
                    frame.position = info["link_positions"][link_idx]
                    frame.wxyz = info["link_wxyzs"][link_idx]
                    color = clearance_color(float(link_min[link_idx]), margin)
                    # Only push a color update when it actually changed; this is
                    # otherwise ~31 extra websocket messages per tick.
                    if capsule_colors[link_idx] != color:
                        capsule_meshes[link_idx].color = color
                        capsule_colors[link_idx] = color
            else:
                for frame in capsule_frames.values():
                    frame.visible = False

            self_distances = info["self_distances"]
            order = np.argsort(self_distances)[:NUM_CLOSEST_PAIRS]
            if show_pair_lines.value:
                centers = info["capsule_centers"]
                closest_pair_segments.points = np.stack(
                    [
                        np.stack(
                            [
                                centers[robot_coll.active_idx_i[k]],
                                centers[robot_coll.active_idx_j[k]],
                            ]
                        )
                        for k in order
                    ]
                ).astype(np.float32)
                closest_pair_segments.colors = np.array(
                    [[clearance_color(float(self_distances[k]), margin)] * 2 for k in order],
                    dtype=np.uint8,
                )
                closest_pair_segments.visible = True
            else:
                closest_pair_segments.visible = False

            min_self = float(self_distances.min())
            min_world = float(info["link_world_min"].min())
            min_self_readout.value = round(min_self, 4)
            min_world_readout.value = round(min_world, 4)
            in_collision_readout.value = int((self_distances < margin).sum())
            closest_pairs_text.content = "\n".join(
                f"- `{self_distances[k]:+.3f}` "
                f"{robot_coll.link_names[robot_coll.active_idx_i[k]]} / "
                f"{robot_coll.link_names[robot_coll.active_idx_j[k]]}"
                for k in order
            )

            for i in range(NUM_OBSTACLES):
                enabled = obstacle_enables[i].value
                obstacle_controls[i].visible = enabled
                obstacle_meshes[i].visible = enabled
                if enabled:
                    obstacle_meshes[i].radius = float(obstacle_radii[i].value)
            ground_plane.visible = show_ground_plane.value
            ground_plane.position = (0.0, 0.0, float(ground_height.value))

            # ---- manipulability ----
            manipulability = info["manipulability"]
            if show_ellipsoids.value:
                scale = float(ellipsoid_scale.value)
                for arm_idx, arm in enumerate(ARMS):
                    # Columns of U are the ellipsoid axes; singular values are
                    # the semi-axis lengths (their product is the Yoshikawa index).
                    axes = info["ellipsoid_axes"][arm_idx]
                    radii = info["singular_values"][arm_idx] * scale
                    vertices = (ellipsoid_vertices * radii) @ axes.T + info[
                        "ee_positions"
                    ][arm_idx]
                    ellipsoid_meshes[arm_idx].vertices = vertices.astype(np.float32)
                    ellipsoid_meshes[arm_idx].visible = True
                    ellipsoid_meshes[arm_idx].color = (
                        manipulability_color(float(manipulability[arm_idx]))
                        if color_by_manip.value
                        else arm.color
                    )
            else:
                for mesh_handle in ellipsoid_meshes:
                    mesh_handle.visible = False

            for arm_idx in range(NUM_ARMS):
                manip_readouts[arm_idx].value = round(float(manipulability[arm_idx]), 5)
                condition_readouts[arm_idx].value = round(
                    float(info["condition"][arm_idx]), 2
                )

            # ---- joint limits ----
            normalized = np.abs(cfg - center) / half_range
            worst_limit_readout.value = round(float(normalized.max()), 3)
            for joint_idx, slider in joint_sliders:
                slider.value = float(
                    np.clip(cfg[joint_idx], lower[joint_idx], upper[joint_idx])
                )

            # ---- cost breakdown ----
            # Mirrors the residual definitions above, so you can see which term
            # is actually driving the solution rather than guessing.
            enables = np.array([float(h.value) for h in arm_enables])
            pos_w = np.array([h.value for h in position_weights]) * enables
            manip_w = np.array([h.value for h in manipulability_weights])
            cost_readouts["pose"].value = round(
                float(np.sum((pos_w * position_errors) ** 2)), 4
            )
            cost_readouts["manipulability"].value = round(
                float(np.sum((manip_w * MANIP_REFERENCE / (manipulability + EPS)) ** 2)), 4
            )
            cost_readouts["centering"].value = round(
                float(
                    np.sum(
                        (
                            centering_weight.value
                            * centering_mask
                            * (cfg - center)
                            / half_range
                        )
                        ** 2
                    )
                ),
                4,
            )
            cost_readouts["barrier"].value = round(
                float(
                    np.sum(
                        (
                            barrier_weight.value
                            * centering_mask
                            * np.maximum(0.0, normalized - float(barrier_activation.value))
                        )
                        ** 2
                    )
                ),
                4,
            )
            cost_readouts["nominal"].value = round(
                float(
                    np.sum((nominal_weight.value * centering_mask * (cfg - nominal_q)) ** 2)
                ),
                4,
            )
            self_violation = np.maximum(0.0, margin - self_distances)
            cost_readouts["self collision"].value = round(
                float(np.sum((self_collision_weight.value * self_violation) ** 2)), 4
            )
            world_violation = np.maximum(0.0, margin - info["link_world_min"])
            cost_readouts["world collision"].value = round(
                float(np.sum((world_collision_weight.value * world_violation) ** 2)), 4
            )
            cost_readouts["solver total"].value = round(float(total_cost), 4)

            # ---- plots + timing ----
            elapsed = loop_start - start_time
            time_history.push(elapsed)
            manip_history.push(manipulability)
            error_history.push(position_errors * 1e3)
            clearance_history.push([min_self, min_world])
            solve_time_history.push([solve_ms, per_iteration_ms])

            times = time_history.values[:, 0].astype(np.float64)
            if times.shape[0] > 2:
                manip_plot.data = (times, *manip_history.values.astype(np.float64).T)
                error_plot.data = (times, *error_history.values.astype(np.float64).T)
                clearance_plot.data = (
                    times,
                    *clearance_history.values.astype(np.float64).T,
                )
                solve_time_plot.data = (
                    times,
                    *solve_time_history.values.astype(np.float64).T,
                )

            # ---- timing readouts ----
            recent_solve_ms = solve_time_history.values[-TIMING_WINDOW:, 0]
            solve_time_readout.value = round(solve_ms, 3)
            solve_avg_readout.value = round(float(recent_solve_ms.mean()), 3)
            solve_max_readout.value = round(float(recent_solve_ms.max()), 3)
            iterations_readout.value = iterations
            per_iteration_readout.value = round(per_iteration_ms, 3)
            diagnostics_time_readout.value = round(
                (time.perf_counter() - draw_start) * 1e3, 3
            )
            total_cost_readout.value = round(float(total_cost), 4)
            now = time.perf_counter()
            loop_time_readout.value = round((now - loop_start) * 1e3, 3)
            loop_rate_readout.value = round(1.0 / max(now - last_loop, 1e-6), 1)
            last_loop = now

            remaining = dt - (now - loop_start)
            if remaining < 0.0:
                slow_counter += 1
                if slow_counter % 100 == 1:
                    print(
                        f"Loop is behind the {rate_slider.value} Hz target "
                        f"(solve took {solve_ms:.1f} ms). Lower the rate or the "
                        f"solver iteration count."
                    )
            time.sleep(max(0.0, remaining))
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
