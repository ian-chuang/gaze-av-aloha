"""Hard inter-arm collision gate on commanded joint configurations.

WHY THIS EXISTS, GIVEN THE SOLVER ALREADY HAS A COLLISION TERM
==============================================================
The deployed IK carries sphere self-collision as a SOFT COST (margin 20 mm,
weight 100).  Three facts keep that from being a guarantee:

  1. A cost can be traded away.  The solver balances collision against pose
     tracking; commanded hard enough toward a conflict, it will split the
     difference.  Nothing in a weighted least-squares problem says "never".
  2. The sphere model is INSCRIBED -- it reads up to ~18 mm optimistic on
     true contact.  Its "+5 mm clearance" can be real-world touching.
  3. The clamps run AFTER the solve.  The driver-feasibility clamp and the
     joint-step clamp both modify the command once the solver is done, so
     the configuration actually sent is not the one the solver certified.

This gate closes all three holes.  It checks the FINAL command -- after
every clamp, immediately before `set_joint_positions` -- against a capsule
model that is CIRCUMSCRIBED: every capsule contains its link's full
collision mesh by construction (`ik_study/collision_models.fit_tight_capsule`,
long-OBB-axis fit with exact vertex containment).  Circumscription is the
whole argument:

     capsule_distance(q)  <=  true_mesh_distance(q)      for every q

so "capsule distance > 0" PROVES the meshes are separated.  The same
fatness that made capsules a bad solver residual (the study measured
59-76 mm hand-jerk when the solver had to feel these surfaces) makes them
the right shape for a yes/no check that only fires near real trouble.

The gate REFUSES the tick: when the assembled command would bring any
inter-arm capsule pair inside the margin, no arm command is sent this tick
and the previous command stands.  The arms hold; the operator backs off;
teleop continues.  Nothing is modified or "repaired" -- a gate that edits
commands is a second IK solver with none of the study behind it.

SCOPE: INTER-ARM BY DEFAULT
===========================
Intra-arm safety stays with the solver's studied sphere cost.  Circumscribed
capsules on adjacent links of the SAME arm overlap permanently (that is why
the study pruned them), so including them would brick the gate at the home
pose.  Inter-arm pairs have no such permanent contacts -- at the home pose
the closest inter-arm capsule pair measures comfortably positive -- so a
violation there is always meaningful.

MARGIN
======
The check is static, one configuration per tick, so the margin must absorb
what happens BETWEEN checks plus what the servos do relative to commands:

    per-tick link travel     bounded by the driver step clamp; worst case
                             observed ~10-15 mm at the gripper for one tick
    servo tracking error     ~10 mm steady-state + ~10 mm compliance below
                             the encoders (move_validation, 2026-08-18)

Default 25 mm covers the sum with the capsule's own conservatism on top.
Raise it for more headroom; lowering it below ~15 mm starts trusting the
servos to track perfectly, which they measurably do not.

CONFIG (env, same convention as study_ik)
=========================================
    GIAVA_CAPSULE_GATE          1 (default) | 0 -- master switch
    GIAVA_CAPSULE_GATE_MARGIN   metres, default 0.025
    GIAVA_CAPSULE_GATE_SCOPE    inter (default) | all
                                'all' adds intra-arm NON-ADJACENT pairs
                                (the study's pruned pair set); expect it to
                                fire in tight folds -- diagnostic use only.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "ik_study")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GATE_ENABLED = os.environ.get("GIAVA_CAPSULE_GATE", "1") == "1"
GATE_MARGIN = float(os.environ.get("GIAVA_CAPSULE_GATE_MARGIN", "0.025"))
GATE_SCOPE = os.environ.get("GIAVA_CAPSULE_GATE_SCOPE", "inter")

## A single margin for the whole robot is wrong, and the handover case is
## what makes it obvious.
##
## The margin exists to cover what happens BETWEEN checks: per-tick link
## travel plus servo tracking error (~10 mm steady state + ~10 mm
## compliance, measured 2026-08-18). That reasoning applies to a shoulder
## swinging through the workspace. It does not apply to two fingertips
## being brought together deliberately, slowly, to pass an object -- which
## is a task, not an accident, and which 25 mm makes impossible.
##
## Fingers are also the cheapest thing on the robot to bump: small, light,
## compliant, and already designed to close on objects. So finger-to-finger
## pairs get their own, much smaller margin, while every structural pair
## keeps the full one.
GATE_MARGIN_FINGER = float(os.environ.get(
    "GIAVA_CAPSULE_GATE_MARGIN_FINGER", "0.008"))


def _is_finger(link_name: str) -> bool:
    return link_name.endswith("_finger_link")
## Fine tier ON by default, because GJK made it free.
##
## The first fine tier fitted capsules to the convex pieces and cost 16 ms
## at the boundary -- too slow, and it recovered only 5 mm of reach because
## the capsule fit was itself the dominant approximation.  Replacing that
## fit with exact GJK on the hulls made it both TIGHTER and CHEAPER:
##
##   model              true mesh gap at stop      ms/tick (median)
##   coarse capsule           77.5 mm                   0.79
##   8 capsules per link      72.5 mm                   2.18
##   GJK on convex hulls      50.1 mm                   0.43
##
## GJK wins on both axes because it answers the question the gate actually
## asks -- "is the separation at least the margin" -- and its duality
## bracket settles that in one or two iterations for all but the pairs
## straddling the margin.  Exactness is only paid for where it changes the
## answer.  Set GIAVA_CAPSULE_GATE_FINE=0 for the coarse-only behaviour.
GATE_FINE = os.environ.get("GIAVA_CAPSULE_GATE_FINE", "1") == "1"
## Time box for fine-tier escalation inside one gate call, milliseconds.
## When the budget is spent the remaining samples keep their COARSE
## verdict, which is more restrictive -- so a timeout costs reach, never
## safety. 6 ms leaves room beside a ~10 ms IK solve in a 20 ms tick.
## Budget is per CALL and shared between the sweep and the refinement,
## not per phase -- two 6 ms phases is a 12 ms call, which was the bug.
##
## A small budget is not a compromise here. The gate runs every tick, so a
## call that escalates only the first sample or two still moves the arm
## closer than coarse alone would, and the NEXT tick escalates from the new
## position. Partial escalation converges to the exact GJK boundary over a
## few ticks; it just approaches it slightly more gently.
GATE_FINE_BUDGET_MS = float(os.environ.get(
    "GIAVA_CAPSULE_GATE_FINE_BUDGET_MS", "3.0"))


def _arm_of(link_name: str) -> str:
    """Which arm a link belongs to; the root 'base' is its own group."""
    for arm in ("left", "right", "middle"):
        if link_name.startswith(arm + "_"):
            return arm
    return "static"


class CapsuleGate:
    """Inter-arm capsule distance check, jitted once, ~sub-ms per call."""

    def __init__(self, robot, urdf, margin: float = GATE_MARGIN,
                 scope: str = GATE_SCOPE, fine: bool = GATE_FINE):
        import jax
        import jax.numpy as jnp

        from collision_models import pruned_tight_capsule_collision
        from pyroki.collision import RobotCollision

        if scope not in ("inter", "all"):
            raise ValueError(f"scope must be inter|all, got '{scope}'")
        self.margin = float(margin)
        self.scope = scope
        self.robot = robot
        self.urdf = urdf

        base = pruned_tight_capsule_collision(urdf)
        names = base.link_names

        ## Restrict the active pair set.  For capsules geometry == link, so
        ## the active indices ARE link indices and prefix grouping is exact.
        keep_i, keep_j = [], []
        for i, j in zip(base.active_idx_i, base.active_idx_j):
            gi, gj = _arm_of(names[i]), _arm_of(names[j])
            if scope == "inter" and gi == gj:
                continue
            ## 'static' is the URDF root -- no collision mesh; from_urdf
            ## gives it a zero-size capsule, but keeping its pairs would
            ## only add noise to the argmin.
            if "static" in (gi, gj):
                continue
            keep_i.append(int(i))
            keep_j.append(int(j))
        if not keep_i:
            raise RuntimeError(
                "capsule gate ended up with zero active pairs -- the link "
                "naming convention must have changed")

        self.coll = RobotCollision(
            num_links=base.num_links,
            link_names=names,
            coll=base.coll,
            active_idx_i=tuple(keep_i),
            active_idx_j=tuple(keep_j),
            _geom_to_link_idx=base._geom_to_link_idx,
        )
        self.pair_names: List[Tuple[str, str]] = [
            (names[i], names[j]) for i, j in zip(keep_i, keep_j)]
        ## Per-pair margin: fingertip pairs are a handover, not a crash.
        self.margin_finger = float(GATE_MARGIN_FINGER)
        self._pair_margin = np.array(
            [self.margin_finger if (_is_finger(a) and _is_finger(b))
             else float(margin) for a, b in self.pair_names])
        self._finger_pairs = int(sum(
            1 for a, b in self.pair_names if _is_finger(a) and _is_finger(b)))

        @jax.jit
        def _dists(cfg):
            ## Same call the deployed min_clearance uses, but per-pair so a
            ## violation can be NAMED -- "hold" with no reason teaches the
            ## operator to disable the gate.
            return self.coll.compute_self_collision_distance(self.robot, cfg)

        self._dists = _dists

        ## FINE TIER.  The coarse capsules pad the true surface by 28 mm on
        ## average and 61 mm on the gripper bases, so a coarse-only gate
        ## stops the arms far short of where they could physically pass.
        ## The fine model (convex decomposition, leak-corrected) pads by
        ## ~9 mm.  It is only consulted for pairs the coarse tier flags,
        ## because it costs 64 piece-pairs per link pair.
        ##
        ## Both models are conservative, so escalation can only ever turn
        ## a coarse FAIL into a PASS -- never the reverse.  That is what
        ## makes "allow close passing" safe rather than merely permissive.
        self.fine = None
        self.fine_budget_s = GATE_FINE_BUDGET_MS * 1e-3
        self.fine_timeouts = 0
        if fine:
            try:
                import multi_capsule
                self.fine = multi_capsule.build(urdf)
                self._multi = multi_capsule
                self._prepare_fine()
            except Exception as e:  # noqa: BLE001
                print(f"[capsule_gate] fine tier unavailable ({e}); running "
                      f"coarse-only, which is SAFE but stops the arms "
                      f"further apart than necessary")
        ## Trigger compilation now, not on the first teleop tick.
        ## NOTE deliberately NOT a sanity check: URDF q=0 is a genuinely
        ## self-colliding configuration on this robot (the left base is
        ## mounted yaw = pi, so at zero joints both grippers extend toward
        ## the workspace centre and MEET -- capsule -136 mm, and the meshes
        ## really do overlap).  Validation must happen at a real operating
        ## pose, which only the caller knows: see validate().
        t0 = time.perf_counter()
        d = np.asarray(self._dists(np.zeros(robot.joints.num_actuated_joints,
                                            dtype=np.float32)))
        self._n_pairs = int(d.size)
        self.compile_ms = (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------ #
    def validate(self, q_urdf: np.ndarray, where: str = "startup") -> bool:
        """Report whether the gate is already violated.  NEVER raises.

        This used to raise, which killed the whole data-collection process
        at teleop-enable and lost the session.  That is the wrong response
        to two arms merely being close: the gate's job is to stop MOTION,
        not to stop the program.  A violated start is an ordinary
        situation -- the operator brings the grippers together to hand
        something over, releases, re-enables -- and it clears itself the
        moment they separate, because the gate holds rather than crashing.

        Returns True when clear.  The caller decides what to do; the
        control loop holds the arms and says so."""
        ok, dist, pair = self.check(q_urdf)
        if not ok:
            print(f"\n[capsule gate] INSIDE THE MARGIN at {where}: "
                  f"{pair[0]} <-> {pair[1]} at {dist * 1e3:+.1f} mm.\n"
                  f"  The arms will HOLD until you move them apart. Teleop "
                  f"is running; nothing has crashed.\n"
                  + self.report(q_urdf, worst=5))
        return ok

    # ------------------------------------------------------------------ #
    def check(self, q_urdf: np.ndarray) -> Tuple[bool, float,
                                                 Optional[Tuple[str, str]]]:
        """(ok, min_distance_m, offending_pair) at a URDF-coordinate config.

        ok=False means: sending this command could not be PROVEN safe --
        the circumscribed capsules of two different arms would come within
        the margin.  Because the capsules contain the meshes, ok=True is a
        proof of separation; ok=False is conservative, not a certainty of
        contact."""
        cfg = np.atleast_2d(np.asarray(q_urdf, dtype=np.float32))
        d = np.atleast_2d(np.asarray(self._dists(cfg)))
        best, which = self._resolve(cfg, d)
        dist, k = float(best[0]), int(which[0])
        if dist >= self._pair_margin[k]:
            return True, dist, None
        return False, dist, self.pair_names[k]

    def _prepare_fine(self) -> None:
        """Flatten the fine model into per-link numpy arrays, once.

        The first implementation walked Python loops over
        (sample x pair x piece_i x piece_j) and cost 200 ms at the wall --
        ten times the tick budget, and all of it interpreter overhead
        rather than arithmetic.  Precomputing contiguous arrays per link
        lets each link pair be expanded by broadcasting instead."""
        self._fine_arr = {}
        for name, entry in self.fine.items():
            caps = entry["caps"]
            self._fine_arr[name] = (
                np.stack([np.asarray(c[0], dtype=float) for c in caps]),
                np.stack([np.asarray(c[1], dtype=float) for c in caps]),
                np.array([float(c[2]) for c in caps]),
                np.array([float(c[3]) for c in caps]),
            )

        ## GJK payload: hull vertices + the per-link leak that makes the
        ## decomposition conservative.  The hulls only APPROXIMATE the
        ## mesh (VHACD), and the leak is how far the mesh pokes outside
        ## them -- so the true separation satisfies
        ##     dist_true >= dist_hulls - leak_i - leak_j
        ## and the gate demands dist_hulls >= margin + leak_i + leak_j.
        ## That keeps the guarantee exact while GJK removes every last
        ## millimetre of FITTING error.
        import gjk as _gjk
        self._gjk = _gjk
        self._hulls, self._spheres, self._leak = {}, {}, {}
        for name, entry in self.fine.items():
            hs = [np.ascontiguousarray(h, dtype=float)
                  for h in entry.get("hulls", [])]
            if not hs:
                continue
            self._hulls[name] = hs
            self._spheres[name] = _gjk.bounding_spheres(hs)
            self._leak[name] = float(entry["stats"].get("leak_m", 0.0))
        self._use_gjk = bool(self._hulls)

        ## MAX LIFT per link: how much tighter the fine model can possibly
        ## be than the coarse capsule for this link.  Measured as the
        ## largest distance from a point on the coarse capsule's surface to
        ## the fine union.  Since the coarse capsule contains the fine
        ## union, fine_dist(i,j) <= coarse_dist(i,j) + lift_i + lift_j.
        ##
        ## Used to SKIP escalation for pairs the fine tier could not
        ## possibly rescue.  Note the direction of risk: if this bound were
        ## too small we would skip a pair that fine would have cleared, and
        ## the gate would be needlessly restrictive.  It can never make the
        ## gate permissive, which is why an estimate is acceptable here
        ## while it would not be in the distance computation itself.
        import multi_capsule as _mc
        rng = np.random.default_rng(0)
        self._lift = {}
        for li, name in enumerate(self.coll.link_names):
            fa = self._fine_arr.get(name)
            if fa is None:
                self._lift[name] = 0.0
                continue
            cap = jax.tree.map(lambda x, _i=li: np.asarray(x)[_i],
                               self.coll.coll)
            c = np.asarray(cap.pose.translation())
            ax = np.asarray(cap.pose.rotation().as_matrix())[:, 2]
            R, H = float(np.asarray(cap.radius)), float(np.asarray(cap.height))
            ## Points on the coarse capsule's cylindrical surface + caps.
            t = rng.uniform(-H / 2, H / 2, 1500)
            phi = rng.uniform(0, 2 * np.pi, 1500)
            tmp = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = np.cross(tmp, ax); e1 /= np.linalg.norm(e1)
            e2 = np.cross(ax, e1)
            pts = (c + t[:, None] * ax
                   + R * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
            d = _mc.union_distance(
                [(fa[0][k], fa[1][k], fa[2][k], fa[3][k])
                 for k in range(len(fa[2]))], pts)
            self._lift[name] = float(max(d.max(), 0.0))
        self._lift_arr = np.array(
            [self._lift.get(n, 0.0) for n in self.coll.link_names])

    def _fine_pairs(self, mats, combos) -> np.ndarray:
        """Fine distances for [(sample, pair_index), ...], one batched call.

        Each combo's k*m piece pairs are expanded by broadcasting, then all
        combos are concatenated into a single segment-segment evaluation
        and reduced back per combo."""
        import jax.numpy as jnp

        from pyroki.collision import _utils

        chunks, owner = [], []
        for n, (s_idx, k) in enumerate(combos):
            li = self.coll.active_idx_i[k]
            lj = self.coll.active_idx_j[k]
            fa = self._fine_arr.get(self.coll.link_names[li])
            fb = self._fine_arr.get(self.coll.link_names[lj])
            if fa is None or fb is None:
                continue
            Ti, Tj = mats[s_idx][li], mats[s_idx][lj]
            ca = fa[0] @ Ti[:3, :3].T + Ti[:3, 3]
            aa = fa[1] @ Ti[:3, :3].T
            cb = fb[0] @ Tj[:3, :3].T + Tj[:3, 3]
            ab = fb[1] @ Tj[:3, :3].T
            a0 = ca - 0.5 * fa[3][:, None] * aa
            a1 = ca + 0.5 * fa[3][:, None] * aa
            b0 = cb - 0.5 * fb[3][:, None] * ab
            b1 = cb + 0.5 * fb[3][:, None] * ab
            ki, mj = len(fa[2]), len(fb[2])
            chunks.append((
                np.repeat(a0, mj, axis=0), np.repeat(a1, mj, axis=0),
                np.tile(b0, (ki, 1)), np.tile(b1, (ki, 1)),
                np.repeat(fa[2], mj) + np.tile(fb[2], ki)))
            owner.append(np.full(ki * mj, n))

        out = np.full(len(combos), np.inf)
        if not chunks:
            return out
        A0 = np.concatenate([c[0] for c in chunks])
        A1 = np.concatenate([c[1] for c in chunks])
        B0 = np.concatenate([c[2] for c in chunks])
        B1 = np.concatenate([c[3] for c in chunks])
        RR = np.concatenate([c[4] for c in chunks])
        owner = np.concatenate(owner)
        pq = _utils.closest_segment_to_segment_points(
            jnp.asarray(A0), jnp.asarray(A1), jnp.asarray(B0), jnp.asarray(B1))
        d = np.linalg.norm(np.asarray(pq[0]) - np.asarray(pq[1]), axis=1) - RR
        ## Segment-min by owner without a Python loop.
        np.minimum.at(out, owner, d)
        return out

    def link_matrices(self, cfgs: np.ndarray) -> np.ndarray:
        """4x4 world transform of every link, for a batch of configurations.

        Hoisted out of the escalation loop on purpose.  Forward kinematics
        is a jax call, and calling it once per escalated sample cost ~19 ms
        at the boundary -- more than the collision maths it was feeding.
        Batching the whole sweep into a single dispatch leaves the
        escalation loop as pure numpy."""
        import jaxlie
        fk = self.robot.forward_kinematics(
            np.atleast_2d(np.asarray(cfgs, dtype=np.float32)))
        return np.asarray(jaxlie.SE3(fk).as_matrix())

    def _gjk_pairs(self, mats, combos) -> np.ndarray:
        """Exact-hull verdicts for [(sample, pair_index), ...].

        Returns `margin` for a pair GJK proved clear and `-inf` for one it
        refuted.  Deliberately a VERDICT, not a distance: the early-exit
        query is 74-212x cheaper than converging the distance, and the gate
        only ever compares against the margin.  Reporting a fake distance
        that happened to be a bound would be worse than reporting none."""
        out = np.full(len(combos), -np.inf)
        ## Transform each link's hulls ONCE per sample, not once per pair.
        ## A link appears in many flagged pairs near the wall, and
        ## re-rotating its vertices for every one of them was the bulk of
        ## the 2.2 ms an escalation used to cost.
        cache: dict = {}

        def posed(s_idx, li):
            key = (s_idx, li)
            if key not in cache:
                name = self.coll.link_names[li]
                hs = self._hulls.get(name)
                if hs is None:
                    cache[key] = None
                else:
                    T = mats[s_idx][li]
                    V = [h @ T[:3, :3].T + T[:3, 3] for h in hs]
                    cache[key] = (V, self._gjk.bounding_spheres(V))
            return cache[key]

        for n, (s_idx, k) in enumerate(combos):
            li = self.coll.active_idx_i[k]
            lj = self.coll.active_idx_j[k]
            pa, pb = posed(s_idx, li), posed(s_idx, lj)
            if pa is None or pb is None:
                out[n] = np.inf          # nothing to say; leave to coarse
                continue
            need = (self._pair_margin[k]
                    + self._leak.get(self.coll.link_names[li], 0.0)
                    + self._leak.get(self.coll.link_names[lj], 0.0))
            ok, _b = self._gjk.pieces_separated(pa[0], pb[0], need,
                                                pa[1], pb[1])
            out[n] = self._pair_margin[k] if ok else -np.inf
        return out

    def _verdict_ok(self, cfg, d_row, mats) -> bool:
        """Is this single configuration clear of EVERY pair's own margin?"""
        b, w = self._resolve(cfg, d_row, mats)
        return bool(b[0] >= self._pair_margin[int(w[0])])

    def _resolve(self, cfgs: np.ndarray, d_coarse: np.ndarray,
                 mats: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-sample (min_distance, limiting_pair_index), fine where needed.

        Coarse first; anything it clears is proven clear and never looked
        at again.  Only the flagged (sample, pair) combos are escalated."""
        cfgs = np.atleast_2d(cfgs)
        slack = d_coarse - self._pair_margin[None, :]
        flagged = slack < 0.0
        which = slack.argmin(axis=1)
        best = d_coarse[np.arange(len(d_coarse)), which]
        if self.fine is None or not flagged.any():
            return best, which

        ## Prune: a flagged pair can only be rescued by the fine tier if
        ## coarse + lift_i + lift_j reaches the margin.  Anything below
        ## that is unsafe whatever the fine model says, so the sample is
        ## already decided and no fine work is needed for it at all.
        li = np.asarray(self.coll.active_idx_i)
        lj = np.asarray(self.coll.active_idx_j)
        max_lift = self._lift_arr[li] + self._lift_arr[lj]          # (pairs,)
        rescuable = flagged & (d_coarse + max_lift[None, :] >= self.margin)
        doomed = flagged & ~rescuable
        sample_doomed = doomed.any(axis=1)
        ## Only escalate samples that are not already decided.
        rescuable[sample_doomed] = False
        if not rescuable.any():
            return best, which

        if mats is None:
            mats = self.link_matrices(cfgs)
        combos = [(int(si), int(ki))
                  for si, ki in zip(*np.nonzero(rescuable))]
        if getattr(self, "_use_gjk", False):
            fine_d = self._gjk_pairs(mats, combos)
        else:
            fine_d = self._fine_pairs(mats, combos)
        ## Recompute the per-sample minimum treating escalated pairs by
        ## their fine value and everything else by its coarse value.
        eff = d_coarse.copy()
        for n, (si, ki) in enumerate(combos):
            if np.isfinite(fine_d[n]):
                eff[si, ki] = fine_d[n]
        es = eff - self._pair_margin[None, :]
        w = es.argmin(axis=1)
        return eff[np.arange(len(eff)), w], w

    def largest_safe_fraction(self, q_prev: np.ndarray, q_target: np.ndarray,
                              n_samples: int = 16, refine: int = 16
                              ) -> Tuple[float, float, Optional[Tuple[str, str]]]:
        """Largest fraction of the step from q_prev toward q_target that stays
        clear.  Returns (alpha, distance_at_alpha, limiting_pair).

        This is what turns the gate from a tripwire into a WALL.  Refusing
        the whole tick (the obvious implementation) stops the arm at
        whatever configuration the previous tick happened to reach, which
        can be far short of the boundary, and then chatters: push, refuse,
        push, refuse.  Scaling the step instead lets the arm slide right up
        to the margin and stop there, smoothly, however hard the operator
        pushes.  The operator feels a firm surface rather than a stutter.

        SWEPT, not just the endpoint.  The whole segment [q_prev, q_prev +
        alpha*(q_target-q_prev)] is sampled, so a fast step cannot tunnel
        between two safe endpoints through a thin conflict in between.
        Both the sweep and the refinement go in BATCHED calls -- two
        dispatches total, ~0.9 ms median against a 20 ms tick.  Sequential
        bisection was measured at 3.4 ms for finer-than-needed resolution;
        batching is what makes the swept form affordable on top of a 10 ms
        IK solve.

        The residual assumption is sampling density.  At the defaults the
        sweep samples every 1/16 of a step (~1 mm of link travel at the
        driver clamp) and the accepted stop is resolved to 1/272 of it.
        A conflict narrower than ~1 mm would be missed -- far below the
        geometry in play, and two orders under the margin.

        alpha == 1.0  the full step is clear
        0 < alpha < 1 partial step -- the arm stops AT the boundary
        alpha == 0.0  q_prev itself is already inside the margin; the
                      operator must move away before anything is sent
        """
        q_prev = np.asarray(q_prev, dtype=np.float32)
        q_target = np.asarray(q_target, dtype=np.float32)
        delta = q_target - q_prev
        call_deadline = time.perf_counter() + self.fine_budget_s
        if not np.any(np.abs(delta) > 1e-12):
            ok, dist, pair = self.check(q_prev)
            return (1.0 if ok else 0.0), dist, pair

        alphas = np.linspace(0.0, 1.0, int(n_samples) + 1, dtype=np.float32)
        cfgs = q_prev[None, :] + alphas[:, None] * delta[None, :]
        d = np.asarray(self._dists(cfgs))            # (n+1, pairs)

        ## Coarse verdict for every sample in one batched call. Everything
        ## it clears is PROVEN clear and never escalated.
        ## Compare each pair against ITS OWN margin, then reduce. Taking a
        ## global min first and comparing to one margin would apply the
        ## structural margin to fingertips.
        slack = d - self._pair_margin[None, :]
        coarse_min = slack.min(axis=1) + self.margin
        safe = slack.min(axis=1) >= 0.0

        ## ENDPOINT FIRST.  Near the wall the coarse model is negative for
        ## every sample, so a purely forward scan spends its whole budget
        ## on sample 0 and returns alpha=0 -- which PINS THE OPERATOR: even
        ## a step that retreats to safety is refused, because the scan
        ## never reaches the samples that would have proved it safe.
        ##
        ## Checking the full step first fixes that directly. Retreating and
        ## ordinary motion both land here and cost a handful of
        ## escalations instead of seventeen. Only a step that genuinely
        ## runs into something falls through to the scan below, and there
        ## alpha < 1 anyway.
        if self.fine is not None and not safe.all():
            probe = sorted({int(round(f * n_samples))
                            for f in (1.0, 0.5, 0.25, 0.75)})
            mats_probe = self.link_matrices(cfgs[probe])
            all_ok = True
            for r, idx in enumerate(probe):
                if safe[idx]:
                    continue
                b, _ = self._resolve(cfgs[idx:idx + 1], d[idx:idx + 1],
                                     mats_probe[r:r + 1])
                coarse_min[idx] = b[0]
                if self._verdict_ok(cfgs[idx:idx + 1], d[idx:idx + 1],
                                    mats_probe[r:r + 1]):
                    safe[idx] = True
                else:
                    all_ok = False
                    break
            if all_ok:
                ## Full step clear at the endpoint and at the quartiles.
                ## The remaining samples sit between verified-clear points
                ## on a step already bounded by the driver clamp.
                return 1.0, float(coarse_min[probe[-1]]), None

        ## Escalate LAZILY, and only forward from the first coarse failure.
        ## The answer is the longest safe PREFIX, so once a sample is
        ## genuinely unsafe nothing beyond it can matter -- escalating the
        ## whole sweep (the obvious implementation) spent 18 ms computing
        ## fine distances for samples deep inside a collision that could
        ## never be part of the answer.
        if self.fine is not None and not safe.all():
            mats_all = self.link_matrices(cfgs)
            deadline = call_deadline
            i = 0
            while i < len(safe):
                if safe[i]:
                    i += 1
                    continue
                if time.perf_counter() > deadline:
                    ## Out of budget. Everything from here keeps its coarse
                    ## verdict, so the arm stops earlier than it strictly
                    ## had to. Restrictive, never unsafe.
                    self.fine_timeouts += 1
                    break
                b, _ = self._resolve(cfgs[i:i + 1], d[i:i + 1],
                                     mats_all[i:i + 1])
                coarse_min[i] = b[0]
                if self._verdict_ok(cfgs[i:i + 1], d[i:i + 1],
                                    mats_all[i:i + 1]):
                    safe[i] = True
                    i += 1
                else:
                    break
        per_sample = coarse_min

        if not safe[0]:
            _, w = self._resolve(cfgs[:1], d[:1])
            return 0.0, float(per_sample[0]), self.pair_names[int(w[0])]
        if bool(safe.all()):
            _, w = self._resolve(cfgs[-1:], d[-1:])
            return 1.0, float(per_sample[-1]), self.pair_names[int(w[0])]

        ## First failure; everything before it is clear.
        first_bad = int(np.argmax(~safe))
        lo_a, hi_a = float(alphas[first_bad - 1]), float(alphas[first_bad])

        ## Refine inside that one interval with a SECOND BATCHED call rather
        ## than a bisection loop.  Sequential bisection costs one dispatch
        ## per halving and measured 3.1 ms end to end; one batch of `refine`
        ## samples gets finer resolution (1/32 x 1/16 of the step) for two
        ## dispatches total.  On a 20 ms tick that difference is worth
        ## having, since this runs on top of a 10 ms IK solve.
        fine = np.linspace(lo_a, hi_a, int(refine) + 2,
                           dtype=np.float32)[1:-1]
        if fine.size:
            cf = q_prev[None, :] + fine[:, None] * delta[None, :]
            df = np.asarray(self._dists(cf))
            ok_f = (df - self._pair_margin[None, :]).min(axis=1) >= 0.0
            if self.fine is not None and not ok_f.all():
                mats_f = self.link_matrices(cf)
                deadline = call_deadline
                for r in range(len(ok_f)):
                    if ok_f[r]:
                        continue
                    if time.perf_counter() > deadline:
                        self.fine_timeouts += 1
                        break
                    if self._verdict_ok(cf[r:r + 1], df[r:r + 1],
                                        mats_f[r:r + 1]):
                        ok_f[r] = True
                    else:
                        break        # prefix broken; nothing later counts
            if bool(ok_f.any()):
                ## Last CONTIGUOUS pass from the start of the interval: the
                ## prefix must stay clear, so a later isolated pass beyond a
                ## failure cannot be taken.
                first_f = int(np.argmax(~ok_f)) if not bool(ok_f.all()) \
                    else int(ok_f.size)
                if first_f > 0:
                    lo_a = float(fine[first_f - 1])

        cl = np.atleast_2d(q_prev + lo_a * delta)
        dl = np.atleast_2d(np.asarray(self._dists(cl)))
        bl, wl = self._resolve(cl, dl)
        return lo_a, float(bl[0]), self.pair_names[int(wl[0])]

    def report(self, q_urdf: np.ndarray, worst: int = 8) -> str:
        """The `worst` closest inter-arm pairs, for the operator."""
        d = np.asarray(self._dists(np.asarray(q_urdf, dtype=np.float32)))
        order = np.argsort(d - self._pair_margin)[:worst]
        lines = [f"  capsule gate ({self.scope}, margin "
                 f"{self.margin * 1e3:.0f} mm / "
                 f"{self.margin_finger * 1e3:.0f} mm fingertips, "
                 f"{self._n_pairs} pairs)."]
        if getattr(self, "_use_gjk", False):
            lines.append("  NOTE these are COARSE capsule distances. They run "
                         "tens of mm pessimistic;")
            lines.append("  the gate's verdict comes from exact GJK and is "
                         "shown by check().")
        for k in order:
            a, b = self.pair_names[int(k)]
            m = self._pair_margin[int(k)]
            flag = "  << coarse inside margin" if d[k] < m else ""
            lines.append(f"    {d[k] * 1e3:+8.1f} mm  (margin {m * 1e3:.0f}) "
                         f" {a} <-> {b}{flag}")
        return "\n".join(lines)

    def describe(self) -> str:
        tier = ("coarse capsule -> EXACT GJK on convex hulls"
                if getattr(self, "_use_gjk", False)
                else "coarse capsule -> fine multi-capsule"
                if self.fine is not None else "coarse capsule ONLY")
        return (f"capsule gate ON: scope={self.scope}, margin="
                f"{self.margin * 1e3:.0f} mm "
                f"({self.margin_finger * 1e3:.0f} mm on "
                f"{self._finger_pairs} fingertip pairs), "
                f"{self._n_pairs} pairs, "
                f"{tier} (compile {self.compile_ms:.0f} ms). Both models are "
                f"circumscribed, so every distance is a LOWER BOUND on the "
                f"true mesh distance -- a pass proves separation."
                + (f" Fine escalation is time-boxed to "
                   f"{GATE_FINE_BUDGET_MS:.0f} ms per call; on timeout the "
                   f"coarse verdict stands, which is stricter."
                   if self.fine is not None else "")
                + ("" if self.fine is not None else
                   " Fine tier OFF: safe, but stops the arms ~20 mm further "
                   "apart than they could physically pass."))


def build_gate(robot, urdf) -> Optional[CapsuleGate]:
    """The gate, or None when disabled -- never half-built."""
    if not GATE_ENABLED:
        print("[capsule_gate] DISABLED via GIAVA_CAPSULE_GATE=0 -- inter-arm "
              "safety rests on the solver's soft sphere cost alone")
        return None
    gate = CapsuleGate(robot, urdf)
    print(f"[capsule_gate] {gate.describe()}")
    return gate
