import os
import argparse
import select
import sys
import time
import traceback
import threading
import queue
from typing import Optional

import numpy as np
import rospy
import torch
from interbotix_xs_modules.arm import InterbotixManipulatorXS

# Env and Constants
from gaze_av_aloha.robot.config import REAL_DT, FPS
from gaze_av_aloha.robot.env_leader_follower import (
    RealEnv, 
    get_master_bot_action, 
    reset_master_arm, 
    wait_for_user, 
    reset_env as reset_puppet_env
)

# Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
try:
    from sound_play.libsoundplay import SoundClient
except ImportError:
    SoundClient = None

# Global sound client reused between resets
_sound_client: Optional[SoundClient] = None


def play_reset_sound():
    """Play an audible cue when reset completes; fall back to a terminal bell."""
    global _sound_client
    if SoundClient is None:
        print("\a", end="", flush=True)
        return
    try:
        if _sound_client is None:
            _sound_client = SoundClient(blocking=False)
            time.sleep(0.1)
        _sound_client.say("reset")
    except Exception:
        print("\a", end="", flush=True)


class BackgroundUploader:
    """Upload dataset batches without blocking the main recording loop."""

    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, batch_start: int, batch_end: int):
        """Queue an upload for a batch (for logging only)."""
        self.queue.put((batch_start, batch_end))

    def flush(self):
        """Wait for all queued uploads to finish."""
        self.queue.join()

    def close(self):
        """Stop the worker thread."""
        self.queue.put(None)
        self.queue.join()
        self.thread.join()

    def _worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            batch_start, batch_end = task
            try:
                print(f"[Uploader] Uploading episodes {batch_start} to {batch_end}...")
                self.dataset.push_to_hub()
                print(f"[Uploader] Upload complete for episodes {batch_start} to {batch_end}.")
            except Exception as e:
                print(f"[Uploader] Upload failed for episodes {batch_start} to {batch_end}: {e}")
            finally:
                self.queue.task_done()


def _user_requested_stop():
    """Return True if user pressed Enter in the terminal."""
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline()
        return True
    return False


def run_episode(dataset: LeRobotDataset, env: RealEnv, master_bot_right, episode_idx: int, task: str):
    """
    Runs a single data recording episode using Leader-Follower teleoperation.
    """
    # 1. Reset Puppet and Master
    reset_puppet_env(env, master_bot_right)
    reset_master_arm(master_bot_right)
    play_reset_sound()

    # 2. Wait for user to trigger start (Close Master Gripper)
    print(f"Episode {episode_idx}: Close RIGHT master gripper to start recording.")
    print("Press Enter in this terminal to stop the episode when you are done.")
    wait_for_user(master_bot_right) 

    # 3. Start Recording Loop
    print(f"Starting episode {episode_idx}...")

    step_idx = 0
    
    # Get initial observation
    obs = env.get_obs()
    
    # Initialize action with current master state
    action = get_master_bot_action(master_bot_right)

    while True:
        step_start = time.time()

        wrist_img = obs['images']['wrist_cam_right']
        overhead_img = obs['images']['overhead_cam']

        # --- Data Collection ---
        # Construct the frame for LeRobotDataset
        # Note: RealEnv (Single Arm) returns 7-dim vectors for state/action
        frame = {
            'action': torch.tensor(action, dtype=torch.float32),
            'observation.state': torch.tensor(obs['joints']['position'], dtype=torch.float32),
            'observation.velocity': torch.tensor(obs['joints']['velocity'], dtype=torch.float32),
            'observation.images.wrist_cam_right': wrist_img,
            'observation.images.overhead_cam': overhead_img,
        }
        dataset.add_frame(frame, task=task)

        # --- Step Environment ---
        # 1. Step the puppet with the CURRENT action
        obs, reward, terminated, truncated, info = env.step(action)

        # 2. Get the NEXT action from the Master arm for the next step
        action = get_master_bot_action(master_bot_right)

        # --- Stop Condition ---
        if _user_requested_stop():
            print("Episode finished by user.")
            break

        # --- Timing ---
        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step)) 

        step_idx += 1
    
    return True

def confirm_episode(episode_idx):
    """Ask the user whether to keep the episode."""
    while True:
        resp = input(f"Episode {episode_idx} complete. Save? [Y/n]: ").strip().lower()
        if resp in ("", "y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")

def main(cfg):
    rospy.init_node("leader_follower", anonymous=True)
    print(f"Starting Leader-Follower recording (Right Arm Only), FPS: {FPS}")
    print(cfg)
    
    # Initialize Master Bot (Right Only)
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s", 
        group_name="arm", 
        gripper_name="gripper",
        robot_name='master_right', 
        init_node=False
    )

    num_cameras = 2 # Right Wrist + Overhead
    dataset_root = os.path.join(cfg['root'], cfg['repo_id'])

    # Create or resume dataset
    if os.path.exists(dataset_root):
        print(f"Dataset exists. Resuming from {dataset_root}")
        dataset = LeRobotDataset(repo_id=cfg['repo_id'], root=dataset_root)
        dataset.start_image_writer(num_threads=num_cameras, num_processes=4 * num_cameras)
    else:
        dataset = LeRobotDataset.create(
            repo_id=cfg['repo_id'],
            root=dataset_root,
            fps=FPS,
            features={
                "observation.images.wrist_cam_right": {
                    "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
                },
                "observation.images.overhead_cam": {
                    "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32", "shape": (7,), "names": None, # 7 DOF
                },
                "observation.velocity": {
                    "dtype": "float32", "shape": (7,), "names": None, # 7 DOF
                },
                "action": {
                    "dtype": "float32", "shape": (7,), "names": None, # 7 DOF
                },
            },
            image_writer_threads=num_cameras,
            image_writer_processes=4 * num_cameras,
        )

    current_episode = dataset.num_episodes
    uploader = BackgroundUploader(dataset)
    print(f"Resuming at episode index {current_episode}.")

    if dataset.num_episodes < cfg['num_episodes']:
        # RealEnv (Right Arm Only)
        env = RealEnv(init_node=False) 

        while True:
            if dataset.num_episodes >= cfg['num_episodes']:
                break

            episode_idx = current_episode
            
            # Run Recording
            ok = run_episode(dataset, env, master_bot_right, episode_idx, cfg['task'])

            if not ok:
                dataset.clear_episode_buffer()
                continue

            # Confirm Save/Discard
            ok = confirm_episode(episode_idx)

            if not ok:
                dataset.clear_episode_buffer()
                print(f"Episode {episode_idx} discarded.")
                continue

            # Save to Disk
            dataset.save_episode()
            print(f"Episode {episode_idx} saved.")
            current_episode += 1

            # Upload to Hugging Face (Optional batching)
            if current_episode % cfg['batch_size'] == 0:
                batch_start = current_episode - cfg['batch_size']
                batch_end = current_episode - 1
                uploader.submit(batch_start, batch_end)
                print(f"Queued upload for episodes {batch_start} to {batch_end}.")

    uploader.flush()
    uploader.close()
    print("Data collection complete.")

if __name__ == "__main__":
    # ROS Setup
    parser = argparse.ArgumentParser(description="Record simulation episodes for AV Aloha (Leader-Follower Right Arm).")
    parser.add_argument("--num-episodes", type=int, default=80, help="Number of episodes to record.")
    parser.add_argument("--repo-id", type=str, default="iantc104/datasets_leader", help="Repository ID for the dataset.")
    parser.add_argument("--root", type=str, default="datasets_leader", help="Root directory for the dataset.")
    parser.add_argument("--task", type=str, default="toothbrush", help="Task name for the dataset.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of episodes to record before uploading.")
    args = parser.parse_args()
    
    args_dict = vars(args)

    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        main(args_dict)
    except Exception as e:
        print(f"An error occured: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down...")
        os._exit(42)
