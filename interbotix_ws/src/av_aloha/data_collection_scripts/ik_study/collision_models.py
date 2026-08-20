"""Collision-study — corrected collision models for the GIAVA robot.

`tight_capsule_collision(urdf)` builds a RobotCollision whose per-link capsule
axes follow the link's *longest* oriented-bounding-box dimension (with exact
vertex containment), instead of pyroki's minimum-bounding-cylinder fit which
minimizes cylinder volume and therefore orients plate/finger-like links along
their *thin* dimension — the caps then add a full radius on both flat sides
(user-diagnosed on the gripper fingers: a 10 cm finger became a ~13 cm-long
fat disc).

This is custom code (flagged); pair logic, distances, and costs remain
pyroki-native and unchanged.

Fit definition (per link, on the URDF collision mesh in link frame):
  axis  â    = longest axis of trimesh.bounding_box_oriented
  center     = midpoint of the vertex span along â
  radius R   = max distance of any vertex from the axis line
  height h   = smallest value such that every vertex satisfies the capsule
               containment condition  |z_i| ≤ h/2 + √(R² − r_i²)
The result contains every mesh vertex by construction (conservative), and is
never *less* tight than pyroki's fit along the two dominant dimensions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyroki.collision import RobotCollision
from pyroki.collision._geometry import Capsule


def fit_tight_capsule(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (center (3,), axis (3,), radius, height) of the containment fit.

    Vertices are augmented with face samples so thin meshes with sparse
    vertices are still covered.
    """
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(mesh.faces) > 0:
        pts = np.vstack([pts, mesh.sample(1024)])

    obb = mesh.bounding_box_oriented
    T = np.asarray(obb.primitive.transform, dtype=np.float64)
    extents = np.asarray(obb.primitive.extents, dtype=np.float64)
    axis = T[:3, int(np.argmax(extents))]
    axis = axis / np.linalg.norm(axis)

    center0 = pts.mean(axis=0)
    z = (pts - center0) @ axis
    z_mid = 0.5 * (z.max() + z.min())
    center = center0 + z_mid * axis
    z = z - z_mid

    radial = pts - center - np.outer(z, axis)
    r = np.linalg.norm(radial, axis=1)
    R = float(r.max()) + 1e-4  # containment + hair of numerical padding

    slack = np.sqrt(np.maximum(R**2 - r**2, 0.0))
    h = float(2.0 * np.maximum(np.abs(z) - slack, 0.0).max())
    return center, axis, R, h


def _capsule_from_fit(center, axis, radius, height) -> Capsule:
    """Build a pyroki Capsule (local z = axis) at `center`."""
    z = np.asarray(axis, dtype=np.float64)
    # any orthonormal frame with z as third column
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rm = np.stack([x, y, z], axis=1)
    se3 = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(jnp.asarray(Rm)), jnp.asarray(center)
    )
    cap = Capsule.from_radius_height(
        position=jnp.zeros(3),
        wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]),
        radius=jnp.asarray(radius),
        height=jnp.asarray(height),
    )
    return cap.transform(se3)


def tight_capsule_collision(urdf, user_ignore_pairs: Tuple[Tuple[str, str], ...] = ()
                            ) -> RobotCollision:
    """RobotCollision identical to `from_urdf` except for the capsule fits."""
    base = RobotCollision.from_urdf(urdf, user_ignore_pairs=user_ignore_pairs)
    caps = []
    for name in base.link_names:
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        if mesh.is_empty:
            caps.append(
                Capsule(pose=jaxlie.SE3.identity(), size=jnp.zeros(2))
            )
            continue
        center, axis, R, h = fit_tight_capsule(mesh)
        caps.append(_capsule_from_fit(center, axis, R, h))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *caps)
    return RobotCollision(
        num_links=base.num_links,
        link_names=base.link_names,
        coll=stacked,
        active_idx_i=base.active_idx_i,
        active_idx_j=base.active_idx_j,
        _geom_to_link_idx=base._geom_to_link_idx,
    )


## --------------------------------------------------------------------- ##
## Multi-sphere model (curobo VOXEL algorithm, mirrored dependency-free)
## --------------------------------------------------------------------- ##
## curobo's voxel_fit_mesh: uniform grid over the mesh bbox (~n cells) →
## keep interior points (SDF) → radius = largest inscribed sphere at that
## centre.  Their SDF runs on Warp; we use trimesh's (convex-hull fallback
## for non-watertight links).  Deviation flagged: same algorithm, no GPU dep.
## Under-coverage between inscribed spheres is expected and absorbed by the
## collision margin; the scorecard quantifies the signed bias.

## Per-link sphere budgets — geometry-driven (more for boxes/plates/forks,
## fewer for cylinders), tuned in Phase 6:
SPHERE_BUDGET = {
    "base_link": 12,        # 0.30×0.20×0.08 plate
    "shoulder_link": 4,
    "upper_arm_link": 6,
    "upper_forearm_link": 8,  # 0.20×0.10×0.04 plate-ish
    "lower_forearm_link": 4,
    "wrist_link": 4,
    "gripper_link": 4,
    "gripper_base": 10,     # the fork
    "right_finger_link": 6,
    "left_finger_link": 6,
}
SPHERE_BUDGET_MIDDLE = {
    "middle_base_link": 12,
    "middle_shoulder_link": 4,
    "middle_upper_arm_link": 6,
    "middle_upper_forearm_link": 4,  # slender cylinder
    "middle_lower_forearm_link": 4,
    "middle_wrist_link": 3,
    "middle_pan_link": 3,
    "middle_camera": 2,
    "middle_camera_body": 4,
    "middle_camera_cover": 6,  # thin plate
}


def _budget_for(link_name: str) -> int:
    if link_name in SPHERE_BUDGET_MIDDLE:
        return SPHERE_BUDGET_MIDDLE[link_name]
    for suffix, n in SPHERE_BUDGET.items():
        if link_name.endswith(suffix):
            return n
    return 0


def _solid_bodies(mesh: trimesh.Trimesh):
    """The solid envelope the link occupies: per-connected-body convex hulls.

    The GIAVA link STLs are mostly thin-shell housings (watertight but only
    12–23% volume fill), so inscribed spheres of the *raw* mesh live in the
    walls.  What collision cares about is the occupied envelope — per-body
    hulls (splitting first keeps separate sub-parts from being bridged).
    Negligible sub-bodies (screws, brackets: hull < 8 cm³ and < 6 cm across)
    are dropped — the neighbouring housing envelope covers them."""
    bodies = mesh.split(only_watertight=False)
    if len(bodies) == 0:
        bodies = [mesh]
    hulls = []
    for b in bodies:
        try:
            h = b.convex_hull
        except Exception:
            continue
        if h.volume <= 1e-9:
            continue
        if h.volume < 8e-6 and float(h.extents.max()) < 0.06:
            continue  # negligible hardware
        hulls.append(h)
    if not hulls:
        hulls = [mesh.convex_hull]
    return hulls


PLATE_THICKNESS = 0.025  # hulls thinner than this get the mid-plane treatment
PLATE_RADIUS = 0.012  # target sphere radius on plates (bounded protrusion)


def _fit_plate(h: trimesh.Trimesh, nb: int):
    """Flat hull: 2-D grid of spheres in the mid-plane.

    Inscribed spheres of a plate can never cover its area (r ≤ thickness/2),
    so plates get r = max(thickness/2, 12 mm) with centers on a grid in the
    two long directions — bounded out-of-plane protrusion (≤ 12 − t/2 mm,
    vs. the capsule fit's 70 mm on the camera cover)."""
    T = np.asarray(h.bounding_box_oriented.primitive.transform)
    E = np.asarray(h.bounding_box_oriented.primitive.extents)
    order = np.argsort(E)  # thin axis first
    r = max(E[order[0]] / 2, PLATE_RADIUS)
    # grid counts in the two long directions, proportional to extent
    e1, e2 = E[order[1]], E[order[2]]
    n2 = max(1, int(round(np.sqrt(nb * e2 / max(e1, 1e-6)))))
    n1 = max(1, nb // n2)
    c_local = np.zeros((n1 * n2, 3))
    axes_local = np.eye(3)
    g1 = np.linspace(-e1 / 2 + r, e1 / 2 - r, n1) if n1 > 1 else np.array([0.0])
    g2 = np.linspace(-e2 / 2 + r, e2 / 2 - r, n2) if n2 > 1 else np.array([0.0])
    k = 0
    for a in g1:
        for b in g2:
            c_local[k, order[1]] = a
            c_local[k, order[2]] = b
            k += 1
    c_local = c_local[:k]
    centers = (T[:3, :3] @ c_local.T).T + T[:3, 3]
    return centers, np.full(len(centers), r)


def fit_spheres_voxel(mesh: trimesh.Trimesh, n: int):
    """curobo voxel-fit (interior grid → inscribed radii) on per-body hulls,
    spread-aware selection, plate-aware fallback.  Exact budget total."""
    if mesh.is_empty or n <= 0:
        return np.zeros((0, 3)), np.zeros(0)
    hulls = _solid_bodies(mesh)
    vols = np.array([max(h.volume, 1e-9) for h in hulls])
    budgets = np.maximum(1, np.round(n * vols / vols.sum()).astype(int))
    while budgets.sum() > n and budgets.max() > 1:
        budgets[np.argmax(budgets)] -= 1

    all_c, all_r = [], []
    for h, nb in zip(hulls, budgets):
        E = np.asarray(h.bounding_box_oriented.primitive.extents)
        if E.min() < PLATE_THICKNESS:
            c, r = _fit_plate(h, nb)
            all_c.append(c)
            all_r.append(r)
            continue
        lo, hi = h.bounds
        extents = np.maximum(hi - lo, 1e-6)
        pitch = float((np.prod(extents) / (6 * nb)) ** (1 / 3))
        axes = [np.linspace(lo[k] + pitch / 2, hi[k] - pitch / 2,
                            max(int(np.ceil(extents[k] / pitch)), 1))
                for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        sd = trimesh.proximity.signed_distance(h, grid)  # positive inside
        inside = sd > 2e-3  # candidates with a non-degenerate radius
        if not inside.any():
            c, r = _fit_plate(h, nb)
            all_c.append(c)
            all_r.append(r)
            continue
        cand_c, cand_r = grid[inside], sd[inside]
        chosen = [int(np.argmax(cand_r))]
        while len(chosen) < min(nb, len(cand_c)):
            d_near = np.min(
                np.linalg.norm(
                    cand_c[:, None, :] - cand_c[chosen][None, :, :], axis=-1
                ),
                axis=1,
            )
            chosen.append(int(np.argmax(d_near + 0.5 * cand_r)))
        all_c.append(cand_c[chosen])
        all_r.append(cand_r[chosen])
    return np.concatenate(all_c), np.concatenate(all_r)


def sphere_decomposition(urdf, cache: Path | None = None) -> dict:
    """{link: {centers, radii}} for every link with geometry (JSON-cached)."""
    import json

    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    base = RobotCollision.from_urdf(urdf)
    out = {}
    for name in base.link_names:
        n = _budget_for(name)
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        centers, radii = fit_spheres_voxel(mesh, n)
        if len(centers):
            out[name] = {"centers": [list(map(float, c)) for c in centers],
                         "radii": [float(r) for r in radii]}
    if cache is not None:
        cache.write_text(json.dumps(out, indent=1))
    return out


def sphere_collision(urdf, cache: Path | None = None) -> RobotCollision:
    """RobotCollision in sphere mode from the (cached) decomposition."""
    decomp = sphere_decomposition(urdf, cache)
    return RobotCollision.from_sphere_decomposition(decomp, urdf)


def pruned_sphere_collision(urdf, results_dir: Path) -> RobotCollision:
    """The deployment model: 180-sphere decomposition with the Phase-7
    corpus-pruned pair set (5,884 of 14,491 sphere pairs kept; functional
    grasping pairs, permanently-inside-margin pairs, and never-active pairs
    removed by rule — see phase7_pair_classify.py)."""
    base = sphere_collision(urdf, results_dir / "sphere_decomposition.json")
    pruning = np.load(results_dir / "sphere_pair_pruning.npz")
    return RobotCollision(
        num_links=base.num_links,
        link_names=base.link_names,
        coll=base.coll,
        active_idx_i=tuple(int(x) for x in pruning["kept_idx_i"]),
        active_idx_j=tuple(int(x) for x in pruning["kept_idx_j"]),
        _geom_to_link_idx=base._geom_to_link_idx,
    )


def pruned_tight_capsule_collision(urdf) -> RobotCollision:
    """Head-to-head capsule reference: corrected fits + the Part-1 link-level
    pruning (12 structural + 2 functional pairs removed)."""
    STRUCTURAL = (
        ("left_base_link", "left_upper_arm_link"),
        ("right_base_link", "right_upper_arm_link"),
        ("middle_base_link", "middle_upper_arm_link"),
        ("middle_camera_body", "middle_camera_cover"),
        ("left_wrist_link", "left_gripper_base"),
        ("right_wrist_link", "right_gripper_base"),
        ("left_lower_forearm_link", "left_gripper_base"),
        ("right_lower_forearm_link", "right_gripper_base"),
        ("middle_pan_link", "middle_camera_cover"),
        ("middle_wrist_link", "middle_camera_cover"),
        ("middle_pan_link", "middle_camera_body"),
        ("middle_lower_forearm_link", "middle_pan_link"),
    )
    FUNCTIONAL = (
        ("left_left_finger_link", "left_right_finger_link"),
        ("right_left_finger_link", "right_right_finger_link"),
    )
    return tight_capsule_collision(urdf, user_ignore_pairs=STRUCTURAL + FUNCTIONAL)


if __name__ == "__main__":
    import robot_model as rm

    robot, urdf = rm.load(with_urdf=True)
    rc_old = RobotCollision.from_urdf(urdf)
    rc_new = tight_capsule_collision(urdf)
    r_old = np.asarray(rc_old.coll.radius)
    h_old = np.asarray(rc_old.coll.height)
    r_new = np.asarray(rc_new.coll.radius)
    h_new = np.asarray(rc_new.coll.height)
    print(f"{'link':28s} {'old r':>6s} {'old h':>6s} {'old len':>8s} | "
          f"{'new r':>6s} {'new h':>6s} {'new len':>8s}")
    for i, name in enumerate(rc_old.link_names):
        if r_old[i] < 1e-6 and r_new[i] < 1e-6:
            continue
        print(
            f"{name:28s} {r_old[i]:6.3f} {h_old[i]:6.3f} "
            f"{h_old[i] + 2 * r_old[i]:8.3f} | "
            f"{r_new[i]:6.3f} {h_new[i]:6.3f} {h_new[i] + 2 * r_new[i]:8.3f}"
        )
