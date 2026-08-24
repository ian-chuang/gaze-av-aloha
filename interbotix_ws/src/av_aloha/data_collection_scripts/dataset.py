import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset

if __package__:
    from .arm_config import ARM_CONFIG
    from .data_col_config import ARM_MODES, DATASET_ROOT
else:
    from arm_config import ARM_CONFIG
    from data_col_config import (
        ARM_MODES,
        DATASET_ROOT,
    )

ARM_DATASET_JOINTS = {
    arm: cfg["joint_names"]
    for arm, cfg in ARM_CONFIG.items()
}


def append_save_debug_line(root: Path, message: str) -> None:
    log_dir = root / "save_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "save_episode.log", "a") as f:
        f.write(f"{message}\n")


# CHANGED (lerobot v0.6.0): this used to be a hand-rolled background saver that
# moved the blocking save_episode() call onto a worker thread, because saving
# meant writing every frame as a PNG and then encoding an MP4 at episode end.
#
# v0.6.0 does this better natively. With `streaming_encoding=True`, frames are fed
# straight to per-camera encoder threads as they are recorded (PyAV releases the
# GIL, so encoding genuinely overlaps the control loop), the PNG round-trip is
# gone entirely, and save_episode() becomes near-instant. That removes the need
# for our own thread, queue and image-writer rotation.
#
# This class is kept only as a thin adapter so the collection scripts keep their
# existing calls. It no longer owns any concurrency of its own.
class BackgroundEpisodeSaver:
    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self.next_episode_index = int(dataset.episode_buffer["episode_index"])
        self._finalized = False

    def save_episode_async(self) -> int:
        if self.dataset.episode_buffer["size"] == 0:
            raise ValueError("No frames available to save.")

        episode_index = int(self.dataset.episode_buffer["episode_index"])

        # Near-instant with streaming encoding: the video is already encoded,
        # this only finalizes the episode and writes its metadata.
        self.dataset.save_episode()

        self.next_episode_index = episode_index + 1
        append_save_debug_line(
            self.dataset.root, f"EPISODE SAVED: episode_{episode_index:04d}"
        )
        return episode_index

    def discard_current_episode(self) -> None:
        # Cancels the in-flight streaming encode and drops the buffer.
        self.dataset.clear_episode_buffer()

    def wait_until_idle(self) -> None:
        # Saving is synchronous now, so there is never outstanding work.
        return

    def close(self) -> None:
        # finalize() flushes buffered episode metadata and writes the parquet
        # footers. Without it the dataset on disk cannot be loaded back.
        #
        # IDEMPOTENT on purpose: this is registered with atexit as well as
        # being called from the normal 'q' shutdown, because a session that
        # ends any other way (exception, Ctrl-C, rospy shutdown) would
        # otherwise leave every episode it recorded unreadable.  Whichever
        # path gets here first does the work; the other returns.
        if self._finalized:
            return
        self._finalized = True
        try:
            self.dataset.finalize()
        except Exception as exc:
            # Reported, not re-raised: this also runs from atexit, where an
            # exception would bury the one line that says what went wrong
            # under an interpreter-shutdown traceback.
            append_save_debug_line(self.dataset.root, f"FINALIZE FAILED: {exc}")
            print(f"[dataset] FINALIZE FAILED: {exc}\n"
                  f"[dataset] {self.dataset.root} may not load back -- "
                  "check it with replay_episode.py --dry-run before recording "
                  "more.")
            return
        append_save_debug_line(self.dataset.root, "DATASET FINALIZED")
        print(f"[dataset] finalized -> {self.dataset.root}")


def build_state_names(active_arms):
    names = []

    for arm in active_arms:
        names.extend(ARM_DATASET_JOINTS[arm])

        if ARM_CONFIG[arm]["has_gripper"]:
            names.append(f"{arm}_gripper")

    return names

def build_action_names(active_arms):
    names = []

    for arm in active_arms:
        names.extend(f"{joint}_cmd" for joint in ARM_DATASET_JOINTS[arm])

        if ARM_CONFIG[arm]["has_gripper"]:
            names.append(f"{arm}_gripper_cmd")

    return names

def add_camera_features(features, active_cameras):

    for camera in active_cameras:

        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }

        # float64, NOT float32.  These are absolute epoch timestamps (~1.79e9
        # right now).  float32 has 24 bits of mantissa, so near that magnitude
        # consecutive representable values are 128 SECONDS apart -- every
        # camera timestamp was being quantised into 128 s buckets, destroying
        # exactly the sub-millisecond information they exist to carry.
        # float64 keeps ~0.2 us at this magnitude.
        features[f"observation.timestamps.{camera}"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": None,
        }

def add_ee_features(features, active_arms):

    for arm in active_arms:

        features[f"observation.ee_pose.{arm}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        }
    
def build_dataset_features(mode, active_cameras):
    active_arms = ARM_MODES[mode]

    state_names = build_state_names(active_arms)
    action_names = build_action_names(active_arms)

    features = {}

    add_camera_features(features, active_cameras)

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": state_names,
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": action_names,
    }

    for arm in active_arms:
        # float64 for the same reason as the camera timestamps above: these
        # are absolute epoch values, where float32 quantises to 128 s.
        features[f"observation.timestamps.{arm}"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": None,
        }

    add_ee_features(features, active_arms)

    return features

def create_dataset(task_name, mode, active_cameras, control_dt):
    repo_id = f"deviamar/{task_name}"

    run_name = time.strftime("%Y%m%d_%H%M%S")

    dataset_root = (
        Path(DATASET_ROOT)
        / task_name
        / run_name
    )

    features = build_dataset_features(mode, active_cameras)

    # CHANGED (lerobot v0.6.0): stream frames to per-camera encoder threads while
    # recording instead of writing PNGs and encoding at episode end. This makes
    # save_episode() near-instant and removes the temp-image round-trip, so the
    # image_writer_* settings are no longer needed.
    #
    # encoder_queue_maxsize is the per-camera frame backlog. If encoding cannot
    # keep up the queue applies back-pressure and frames can be dropped, so this
    # is deliberately generous relative to the default of 30.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_root),
        fps=round(1.0 / control_dt),
        features=features,
        streaming_encoding=True,
        encoder_queue_maxsize=120,
    )

    save_dataset_metadata(
        dataset_root,
        task_name,
        mode,
        active_cameras,
        control_dt,
    )

    return dataset, dataset_root

def get_state_dim(mode):
    return len(build_state_names(ARM_MODES[mode]))

def get_action_dim(mode):
    return len(build_action_names(ARM_MODES[mode]))

def build_frame(
    mode,
    active_cameras,
    robot_states,
    robot_actions,
    ee_poses,
    timestamps,
    images,
):
    active_arms = ARM_MODES[mode]

    frame = {}

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    for camera in active_cameras:

        if camera == "oak_left" or camera == "oak_right":

            frame["observation.images.oak_left"] = torch.from_numpy(
                images["oak_left"]
            )

            frame["observation.images.oak_right"] = torch.from_numpy(
                images["oak_right"]
            )

            frame["observation.timestamps.oak_left"] = torch.tensor(
                [timestamps["oak_left"]],
                dtype=torch.float64,
            )

            frame["observation.timestamps.oak_right"] = torch.tensor(
                [timestamps["oak_right"]],
                dtype=torch.float64,
            )

            continue

        frame[f"observation.images.{camera}"] = torch.from_numpy(
            images[camera]
        )

        frame[f"observation.timestamps.{camera}"] = torch.tensor(
            [timestamps[camera]],
            dtype=torch.float64,
        )

    # ------------------------------------------------------------------
    # Observation state
    # ------------------------------------------------------------------

    observation_state = []

    for arm in active_arms:

        observation_state.extend(
            robot_states[arm]["joints"]
        )

        if ARM_CONFIG[arm]["has_gripper"]:
            observation_state.append(
                robot_states[arm]["gripper"]
            )

    frame["observation.state"] = torch.tensor(
        observation_state,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    action = []

    for arm in active_arms:

        action.extend(
            robot_actions[arm]["joints"]
        )

        if ARM_CONFIG[arm]["has_gripper"]:
            action.append(
                robot_actions[arm]["gripper"]
            )

    frame["action"] = torch.tensor(
        action,
        dtype=torch.float32,
    )

    assert len(observation_state) == get_state_dim(mode), (
        f"Observation dimension mismatch: "
        f"{len(observation_state)} != {get_state_dim(mode)}"
    )

    assert len(action) == get_action_dim(mode), (
        f"Action dimension mismatch: "
        f"{len(action)} != {get_action_dim(mode)}"
    )

    # ------------------------------------------------------------------
    # EE poses
    # ------------------------------------------------------------------

    for arm in active_arms:

        frame[f"observation.ee_pose.{arm}"] = torch.tensor(
            ee_poses[arm],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Robot timestamps
    # ------------------------------------------------------------------

    for arm in active_arms:

        frame[f"observation.timestamps.{arm}"] = torch.tensor(
            [timestamps[arm]],
            dtype=torch.float64,
        )

    return frame

def save_dataset_metadata(dataset_root, task_name, mode, active_cameras, control_dt):
    metadata = {
        "task": task_name,
        "mode": mode,
        "active_arms": ARM_MODES[mode],
        "joint_names": build_state_names(ARM_MODES[mode]),
        "active_cameras": active_cameras,
        "state_dim": get_state_dim(mode),
        "action_dim": get_action_dim(mode),
        "fps": round(1.0 / control_dt),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(dataset_root / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
