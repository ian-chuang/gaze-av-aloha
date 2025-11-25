import os
import argparse
import select
import sys
import time
import traceback
import threading
import queue
import inspect
from typing import Optional, Callable

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
except Exception:
    SoundClient = None

# Global sound client reused between resets
_sound_client: Optional[SoundClient] = None


def _beep(times: int = 2, gap: float = 0.08):
    """Emit a system bell multiple times to make the cue noticeable."""
    for _ in range(times):
        print("\a", end="", flush=True)
        time.sleep(gap)


def play_reset_sound():
    """Play an audible cue when reset completes; combines TTS (if available) and a bell."""
    global _sound_client
    # Try voice if sound_play is available
    if SoundClient is not None:
        try:
            if _sound_client is None:
                _sound_client = SoundClient(blocking=False)
                time.sleep(0.1)
            _sound_client.say("reset")
        except Exception:
            pass
    # Always emit a bell so there is a clear cue
    _beep(times=3, gap=0.07)


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


class BackgroundEpisodeWorker:
    """
    Persist episodes and handle uploads in the background so recording can start immediately.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        dataset_factory: Callable[[], LeRobotDataset],
    ):
        self.dataset = dataset
        self.dataset_factory = dataset_factory
        self.queue: queue.Queue = queue.Queue()
        self._save_supports_buffer = self._detect_buffer_arg(dataset)
        # Use a dedicated dataset handle for saving if we cannot pass buffers directly.
        self._save_dataset = dataset if self._save_supports_buffer else self.dataset_factory()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit_save(self, episode_buffer, episode_idx: int):
        self.queue.put(("save", episode_buffer, episode_idx))

    def submit_upload(self, batch_start: int, batch_end: int):
        self.queue.put(("upload", batch_start, batch_end))

    def flush(self):
        self.queue.join()

    def close(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()

    def _detect_buffer_arg(self, dataset: LeRobotDataset) -> bool:
        try:
            sig = inspect.signature(dataset.save_episode)
            return "episode_buffer" in sig.parameters
        except Exception:
            return False

    def _worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            kind, *payload = task
            try:
                if kind == "save":
                    episode_buffer, episode_idx = payload
                    self._save_episode(episode_buffer, episode_idx)
                elif kind == "upload":
                    batch_start, batch_end = payload
                    print(f"[Uploader] Uploading episodes {batch_start} to {batch_end}...")
                    self._save_dataset.push_to_hub()
                    print(f"[Uploader] Upload complete for episodes {batch_start} to {batch_end}.")
            except Exception as e:
                print(f"[BackgroundWorker] Task {kind} failed: {e}")
            finally:
                self.queue.task_done()

    def _save_episode(self, episode_buffer, episode_idx: int):
        try:
            if self._save_supports_buffer:
                self._save_dataset.save_episode(episode_buffer=episode_buffer)
            else:
                # Fallback for older APIs that rely on episode_buffer on the instance.
                prev_buffer = getattr(self._save_dataset, "episode_buffer", None)
                self._save_dataset.episode_buffer = episode_buffer
                self._save_dataset.save_episode()
                self._save_dataset.episode_buffer = prev_buffer
            print(f"Episode {episode_idx} saved in background.")
        except Exception as e:
            print(f"[BackgroundWorker] Save failed for episode {episode_idx}: {e}")


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
    reset_threads = [
        threading.Thread(target=reset_puppet_env, args=(env, master_bot_right)),
        threading.Thread(target=reset_master_arm, args=(master_bot_right,)),
    ]
    for t in reset_threads:
        t.start()
    for t in reset_threads:
        t.join()
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
            _beep(times=2, gap=0.05)
            return True
        if resp in ("n", "no"):
            _beep(times=1, gap=0.15)
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
    image_writer_config = {"num_threads": num_cameras, "num_processes": 4 * num_cameras}

    # Create or resume dataset
    if os.path.exists(dataset_root):
        print(f"Dataset exists. Resuming from {dataset_root}")
        dataset = LeRobotDataset(repo_id=cfg['repo_id'], root=dataset_root)
        dataset.start_image_writer(**image_writer_config)
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
    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=current_episode)
    print(f"Resuming at episode index {current_episode}.")

    def dataset_factory_for_saving():
        ds = LeRobotDataset(repo_id=cfg['repo_id'], root=dataset_root)
        ds.start_image_writer(**image_writer_config)
        return ds

    background_worker = BackgroundEpisodeWorker(dataset, dataset_factory_for_saving)

    if current_episode < cfg['num_episodes']:
        # RealEnv (Right Arm Only)
        env = RealEnv(init_node=False) 

        while True:
            if current_episode >= cfg['num_episodes']:
                break

            episode_idx = current_episode
            
            # Run Recording
            ok = run_episode(dataset, env, master_bot_right, episode_idx, cfg['task'])

            if not ok:
                dataset.clear_episode_buffer()
                dataset.episode_buffer = dataset.create_episode_buffer(episode_index=current_episode)
                continue

            # Confirm Save/Discard
            ok = confirm_episode(episode_idx)

            if not ok:
                dataset.clear_episode_buffer()
                print(f"Episode {episode_idx} discarded.")
                dataset.episode_buffer = dataset.create_episode_buffer(episode_index=current_episode)
                continue

            # Save to Disk in the background
            episode_buffer_to_save = dataset.episode_buffer
            background_worker.submit_save(episode_buffer_to_save, episode_idx)
            print(f"Episode {episode_idx} queued for saving.")
            current_episode += 1
            dataset.episode_buffer = dataset.create_episode_buffer(episode_index=current_episode)

            # Upload to Hugging Face (Optional batching)
            if current_episode % cfg['batch_size'] == 0:
                batch_start = current_episode - cfg['batch_size']
                batch_end = current_episode - 1
                background_worker.submit_upload(batch_start, batch_end)
                print(f"Queued upload for episodes {batch_start} to {batch_end}.")

    background_worker.flush()
    background_worker.close()
    print("Data collection complete.")

if __name__ == "__main__":
    # ROS Setup
    parser = argparse.ArgumentParser(description="Record simulation episodes for AV Aloha (Leader-Follower Right Arm).")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to record.")
    parser.add_argument("--repo-id", type=str, default="iantc104/store_drawer_lf", help="Repository ID for the dataset.")
    parser.add_argument("--root", type=str, default="datasets_leader/store_drawer_lf", help="Root directory for the dataset.")
    parser.add_argument("--task", type=str, default="store_drawer", help="Task name for the dataset.")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of episodes to record before uploading.")
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
