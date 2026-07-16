import numpy as np
import torch
from pathlib import Path
import time
import json

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from arm_config import ARM_CONFIG
from data_col_config import (
    ARM_MODES,
    DATASET_ROOT,
)

ARM_DATASET_JOINTS = {
    arm: cfg["joint_names"]
    for arm, cfg in ARM_CONFIG.items()
}

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