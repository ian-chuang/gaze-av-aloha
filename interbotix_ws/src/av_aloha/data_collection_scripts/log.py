import logging
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

@dataclass
class SessionStats:
    frames_added: int = 0

    teleop_enable_count: int = 0
    teleop_disable_count: int = 0

    cameras: dict[str, CameraStats] = field(default_factory=dict)
    arms: dict[str, ArmStats] = field(default_factory=dict)

    loop: list[float] = field(default_factory=list)
    headset: list[float] = field(default_factory=list)
    ik_solve: list[float] = field(default_factory=list)
    cmd: list[float] = field(default_factory=list)
    overruns: int = 0

def reset_episode_log(active_cameras, active_arms):
    stats = SessionStats()

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
        f"teleop_enable={episode_stats.teleop_enable_count}",
        f"teleop_disable={episode_stats.teleop_disable_count}",
        f"overruns={episode_stats.overruns}",
        f"loop_ms={1000*_mean(episode_stats.loop):.2f}",
        f"headset_ms={1000*_mean(episode_stats.headset):.2f}",
        f"ik_ms={1000*_mean(episode_stats.ik_solve):.2f}",
        f"cmd_ms={1000*_mean(episode_stats.cmd):.2f}",
    ]

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