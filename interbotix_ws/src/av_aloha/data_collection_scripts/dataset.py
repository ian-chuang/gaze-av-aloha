import numpy as np
import torch
from pathlib import Path
import time
import json
import queue
import shutil
import threading

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, append_save_debug_line

from arm_config import ARM_CONFIG
from data_col_config import (
    ARM_MODES,
    DATASET_ROOT,
)

ARM_DATASET_JOINTS = {
    arm: cfg["joint_names"]
    for arm, cfg in ARM_CONFIG.items()
}


# CHANGED: background saver keeps a single writer for dataset metadata while allowing
# the foreground collector to start the next episode immediately.
class BackgroundEpisodeSaver:
    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self.next_episode_index = dataset.episode_buffer["episode_index"]
        self._save_queue = queue.Queue()
        self._save_error = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while True:
            episode_data = self._save_queue.get()
            if episode_data is None:
                self._save_queue.task_done()
                break

            episode_index = int(episode_data["episode_index"])
            try:
                self.dataset.save_episode(episode_data=episode_data)
                # CHANGED: write background save completion to the save log instead of the terminal.
                append_save_debug_line(
                    self.dataset.root,
                    f"EPISODE SAVED: episode_{episode_index:04d}",
                )
            except Exception as exc:
                self._save_error = exc
                # CHANGED: write background save failures to the save log instead of the terminal.
                append_save_debug_line(
                    self.dataset.root,
                    f"Background episode save failed for episode_{episode_index:04d}: {exc}",
                )
            finally:
                self._save_queue.task_done()
                # CHANGED: print a single terminal message when the background save queue becomes empty.
                if self._save_queue.unfinished_tasks == 0:
                    print("\nBACKGROUND SAVES COMPLETE")

    def _raise_if_save_failed(self):
        if self._save_error is not None:
            raise RuntimeError("Background episode save failed.") from self._save_error

    def _rotate_image_writer(self):
        if self.dataset.image_writer is None:
            return

        # CHANGED: flush only the completed episode's queued PNG writes before
        # handing the old buffer to the background saver.
        num_processes = self.dataset.image_writer.num_processes
        num_threads = self.dataset.image_writer.num_threads
        self.dataset.stop_image_writer()
        self.dataset.start_image_writer(num_processes, num_threads)

    def save_episode_async(self) -> int:
        self._raise_if_save_failed()

        if self.dataset.episode_buffer["size"] == 0:
            raise ValueError("No frames available to save.")

        self._rotate_image_writer()

        episode_data = self.dataset.episode_buffer
        episode_index = episode_data["episode_index"]

        # CHANGED: reserve the next episode index immediately so new image paths
        # do not collide while the previous episode saves in the background.
        self.next_episode_index = episode_index + 1
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.next_episode_index)

        self._save_queue.put(episode_data)
        # CHANGED: write queue events to the save log instead of the terminal.
        append_save_debug_line(
            self.dataset.root,
            f"Queued episode_{episode_index:04d} for background save.",
        )
        return episode_index

    def discard_current_episode(self) -> None:
        self._raise_if_save_failed()

        episode_index = self.dataset.episode_buffer["episode_index"]

        if self.dataset.image_writer is not None:
            for cam_key in self.dataset.meta.camera_keys:
                img_dir = self.dataset._get_image_file_path(
                    episode_index=episode_index,
                    image_key=cam_key,
                    frame_index=0,
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # CHANGED: preserve the reserved episode index when discarding the active buffer.
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(episode_index)

    def wait_until_idle(self) -> None:
        self._save_queue.join()
        self._raise_if_save_failed()

    def close(self) -> None:
        self.wait_until_idle()
        self._save_queue.put(None)
        self._worker.join()

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

        features[f"observation.timestamps.{camera}"] = {
            "dtype": "float32",
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
        features[f"observation.timestamps.{arm}"] = {
            "dtype": "float32",
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

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_root),
        fps=round(1.0 / control_dt),
        features=features,
        image_writer_threads=4,
        image_writer_processes=0,
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
                dtype=torch.float32,
            )

            frame["observation.timestamps.oak_right"] = torch.tensor(
                [timestamps["oak_right"]],
                dtype=torch.float32,
            )

            continue

        frame[f"observation.images.{camera}"] = torch.from_numpy(
            images[camera]
        )

        frame[f"observation.timestamps.{camera}"] = torch.tensor(
            [timestamps[camera]],
            dtype=torch.float32,
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
            dtype=torch.float32,
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
