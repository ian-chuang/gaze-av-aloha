import logging
import os as _os
from dataclasses import dataclass, field
import numpy as np

logging.basicConfig(
    filename="teleop_timing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

@dataclass
class CameraStats:
    frames_missing: int = 0
    timestamps_missing: int = 0
    lag_dt: list[float] = field(default_factory=list)

@dataclass
class ArmStats:
    ik_attempts: int = 0
    ik_successes: int = 0
    ik_failures: int = 0

    ik_position_errors: list[float] = field(default_factory=list)
    ik_orientation_errors: list[float] = field(default_factory=list)

    joint_step_norms: list[float] = field(default_factory=list)

    cmd_track_err: list[float] = field(default_factory=list)
    # Asymmetry discriminators (see log_episode_info): commanded waist angle
    # and target positions per tick.
    waist_cmds: list[float] = field(default_factory=list)
    target_positions: list = field(default_factory=list)

@dataclass
class SessionStats:
    frames_added: int = 0
    ## Frames dropped BEFORE teleop was first enabled in this episode
    ## (GIAVA_RECORD_GATE=teleop).  These are the operator walking from the
    ## keyboard back to the controllers: the arm parked at the reset pose,
    ## action == reset pose, repeated.  Recording them makes the reset pose
    ## the single most common action in the dataset and teaches a policy to
    ## fall back to it whenever the scene looks static.  Counted so the cost
    ## of the habit is visible per episode.
    frames_skipped_pre_teleop: int = 0

    teleop_enable_count: int = 0
    teleop_disable_count: int = 0

    cameras: dict[str, CameraStats] = field(default_factory=dict)
    arms: dict[str, ArmStats] = field(default_factory=dict)

    loop: list[float] = field(default_factory=list)
    headset: list[float] = field(default_factory=list)
    ik_solve: list[float] = field(default_factory=list)
    # Coupled-IK solve time per tick and how often it exceeded the control
    # period (sphere collision is cheap on average but has a long tail).
    ik_solve_ms: list[float] = field(default_factory=list)
    ik_overrun_ticks: int = 0
    # Ticks where the post-solve joint clamp was saturating (only counted when
    # TeleopConfig.enable_joint_clamp is on).
    clamp_saturated_ticks: int = 0
    # Ticks where a command had to be pulled back into the driver's feasible
    # set (position or per-tick velocity), and which joints were responsible.
    driver_clamp_ticks: int = 0
    ## Ticks on which the capsule gate refused the assembled command --
    ## the arms held instead of moving toward an inter-arm contact.
    capsule_gate_blocks: int = 0
    ## Ticks where the gate SHORTENED the step rather than refusing it:
    ## the arms slid up to the margin and stopped there. Common and
    ## healthy near the boundary; blocks are the harsher outcome.
    capsule_gate_scaled: int = 0
    capsule_gate_min_alpha: float = 1.0
    ## Same two counters, for the tabletop floor gate (table_gate.py) --
    ## UNVALIDATED, so worth watching separately from the inter-arm gate.
    table_gate_blocks: int = 0
    table_gate_scaled: int = 0
    table_gate_min_alpha: float = 1.0
    driver_clamp_joints: dict[str, int] = field(default_factory=dict)
    # Residual misalignment of each timestep's camera frames, in seconds:
    # max minus min of the timestamps actually chosen.  Only populated when
    # GIAVA_SYNC_FRAMES is on (camera_manager.select_synchronized_frames).
    # These cameras free-run, so this is the software alignment achieved --
    # it is recorded rather than assumed.
    sync_spreads: list[float] = field(default_factory=list)
    cmd: list[float] = field(default_factory=list)
    overruns: int = 0
    ## Which collision configuration produced this session (collision_modes.py).
    collision_mode: str = "unknown"
    ## Whether tabletop avoidance was on for this session (--table, UNVALIDATED).
    table_mode: str = "unknown"

def reset_episode_log(active_cameras, active_arms):
    stats = SessionStats()
    ## Stamp the collision configuration on every episode, from the
    ## environment collision_modes.select() populated at startup. Done here
    ## rather than at the call sites so no future one can forget: an
    ## episode's feel is uninterpretable without knowing which model
    ## produced it.
    stats.collision_mode = _os.environ.get("GIAVA_COLLISION", "unknown")
    stats.table_mode = _os.environ.get("GIAVA_TABLE", "unknown")

    for camera in active_cameras:
        stats.cameras[camera] = CameraStats()

    for arm in active_arms:
        stats.arms[arm] = ArmStats()

    return stats

def _mean(values):
    return np.mean(values) if values else 0.0

def _max(values):
    return np.max(values) if values else 0.0

def log_episode_info(episode_idx, episode_stats):
    msg = [
        f"episode={episode_idx:04d}",
        f"frames_added={episode_stats.frames_added}",
        f"frames_skipped_pre_teleop="
        f"{getattr(episode_stats, 'frames_skipped_pre_teleop', 0)}",
        f"teleop_enable={episode_stats.teleop_enable_count}",
        f"teleop_disable={episode_stats.teleop_disable_count}",
        f"overruns={episode_stats.overruns}",
        f"loop_ms={1000*_mean(episode_stats.loop):.2f}",
        f"headset_ms={1000*_mean(episode_stats.headset):.2f}",
        f"ik_ms={1000*_mean(episode_stats.ik_solve):.2f}",
        f"cmd_ms={1000*_mean(episode_stats.cmd):.2f}",
    ]

    # Coupled-IK timing detail: the mean hides the tail that actually causes
    # missed ticks, so report p95/max and the overrun count explicitly.
    if episode_stats.ik_solve_ms:
        _ms = sorted(episode_stats.ik_solve_ms)
        _p95 = _ms[min(len(_ms) - 1, int(0.95 * len(_ms)))]
        msg.extend([
            f"ik_solve_mean_ms={_mean(episode_stats.ik_solve_ms):.2f}",
            f"ik_solve_p95_ms={_p95:.2f}",
            f"ik_solve_max_ms={_ms[-1]:.2f}",
            f"ik_overruns={episode_stats.ik_overrun_ticks}"
            f"/{len(episode_stats.ik_solve_ms)}",
        ])
    if episode_stats.clamp_saturated_ticks:
        msg.append(f"clamp_saturated={episode_stats.clamp_saturated_ticks}")
    ## Lead with the collision configuration: an episode's feel is only
    ## interpretable against the model that produced it.
    _cm = getattr(episode_stats, "collision_mode", None)
    if _cm and _cm != "unknown":
        msg.append(f"collision={_cm}")
    if episode_stats.capsule_gate_blocks:
        msg.append(f"capsule_gate_blocks={episode_stats.capsule_gate_blocks}")
    if episode_stats.capsule_gate_scaled:
        msg.append(
            f"capsule_gate_scaled={episode_stats.capsule_gate_scaled}"
            f"(min step {episode_stats.capsule_gate_min_alpha * 100:.0f}%)")
    _tm = getattr(episode_stats, "table_mode", None)
    if _tm and _tm not in ("unknown", "off"):
        msg.append(f"table={_tm}")
    if episode_stats.table_gate_blocks:
        msg.append(f"table_gate_blocks={episode_stats.table_gate_blocks}")
    if episode_stats.table_gate_scaled:
        msg.append(
            f"table_gate_scaled={episode_stats.table_gate_scaled}"
            f"(min step {episode_stats.table_gate_min_alpha * 100:.0f}%)")
    if episode_stats.driver_clamp_ticks:
        worst = sorted(episode_stats.driver_clamp_joints.items(),
                       key=lambda kv: -kv[1])[:3]
        msg.append(f"driver_clamped={episode_stats.driver_clamp_ticks}")
        msg.append("driver_clamp_joints=" + ",".join(f"{k}:{v}" for k, v in worst))

    if episode_stats.sync_spreads:
        sp = sorted(episode_stats.sync_spreads)
        msg.append(
            f"cam_sync_spread_ms={1000*_mean(sp):.2f}"
            f"/p95={1000*sp[min(len(sp)-1, int(0.95*len(sp)))]:.2f}"
            f"/max={1000*sp[-1]:.2f}")

    for cam_name, cam_stats in episode_stats.cameras.items():

        msg.extend([
            f"{cam_name}_missing={cam_stats.frames_missing}",
            f"{cam_name}_missing_ts={cam_stats.timestamps_missing}",
            f"{cam_name}_lag_ms={1000*_mean(cam_stats.lag_dt):.2f}",
        ])

    for arm_name, arm_stats in episode_stats.arms.items():

        success_rate = (
            arm_stats.ik_successes / arm_stats.ik_attempts
            if arm_stats.ik_attempts > 0
            else 0.0
        )

        # Asymmetry discriminators: wild waist range with a quiet target =
        # pose-dependent conditioning (target near the waist axis); wild
        # target = input noise (controller tracking / head-composition leak).
        if arm_stats.waist_cmds:
            _w = arm_stats.waist_cmds
            msg.append(f"{arm_name}_waist_range_deg={np.degrees(max(_w) - min(_w)):.1f}")
        if arm_stats.target_positions:
            _t = np.asarray(arm_stats.target_positions)
            msg.append(f"{arm_name}_target_p2p_mm={np.max(np.ptp(_t, axis=0)) * 1e3:.1f}")

        msg.extend([
            f"{arm_name}_ik_attempts={arm_stats.ik_attempts}",
            f"{arm_name}_ik_successes={arm_stats.ik_successes}",
            f"{arm_name}_ik_failures={arm_stats.ik_failures}",
            f"{arm_name}_ik_success_rate={success_rate:.3f}",

            f"{arm_name}_mean_pos_err={_mean(arm_stats.ik_position_errors):.5f}",
            f"{arm_name}_max_pos_err={_max(arm_stats.ik_position_errors):.5f}",

            f"{arm_name}_mean_ori_err_rad={_mean(arm_stats.ik_orientation_errors):.5f}",
            f"{arm_name}_max_ori_err_rad={_max(arm_stats.ik_orientation_errors):.5f}",

            f"{arm_name}_mean_joint_step={_mean(arm_stats.joint_step_norms):.5f}",
            f"{arm_name}_max_joint_step={_max(arm_stats.joint_step_norms):.5f}",

            f"{arm_name}_mean_track_err={_mean(arm_stats.cmd_track_err):.5f}",
            f"{arm_name}_max_track_err={_max(arm_stats.cmd_track_err):.5f}",
        ])

    logging.info(
        "episode_summary %s",
        " ".join(msg),
    )