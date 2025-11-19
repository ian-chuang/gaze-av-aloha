import numpy as np
import time
import os
import torch
import rospy
import traceback
import argparse
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

# Headset (Only for Feedback/Buttons, not tracking)
from gym_av_aloha.vr.headset import WebRTCHeadset
from gym_av_aloha.vr.headset_utils import HeadsetFeedback

# Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def run_episode(dataset: LeRobotDataset, env: RealEnv, headset: WebRTCHeadset, 
                master_bot_right, episode_idx: int, task: str):
    """
    Runs a single data recording episode using Leader-Follower teleoperation.
    """
    feedback = HeadsetFeedback()
    
    # 1. Reset Puppet and Master
    reset_puppet_env(env, master_bot_right)
    reset_master_arm(master_bot_right)

    # 2. Wait for user to trigger start (Close Master Gripper)
    feedback.info = f"Episode {episode_idx}: Close RIGHT Master Gripper to start."
    headset.send_feedback(feedback)
    
    # This function blocks until user closes the gripper
    wait_for_user(master_bot_right) 

    # 3. Start Recording Loop
    print(f"Starting episode {episode_idx}...")
    feedback.info = f"Recording Episode {episode_idx}... (Press 'A' on headset to Stop)"
    headset.send_feedback(feedback)

    step_idx = 0
    
    # Get initial observation
    obs = env.get_obs()
    
    # Initialize action with current master state
    action = get_master_bot_action(master_bot_right)

    while True:
        step_start = time.time()

        # --- Data Collection ---
        # Construct the frame for LeRobotDataset
        # Note: RealEnv (Single Arm) returns 7-dim vectors for state/action
        frame = {
            'action': torch.tensor(action, dtype=torch.float32),
            'observation.state': torch.tensor(obs['joints']['position'], dtype=torch.float32),
            'observation.velocity': torch.tensor(obs['joints']['velocity'], dtype=torch.float32),
            'observation.images.wrist_cam_right': obs['images']['wrist_cam_right'],
            'observation.images.overhead_cam': obs['images']['overhead_cam'],
        }
        dataset.add_frame(frame, task=task)

        # --- Step Environment ---
        # 1. Step the puppet with the CURRENT action
        obs, reward, terminated, truncated, info = env.step(action)

        # 2. Get the NEXT action from the Master arm for the next step
        action = get_master_bot_action(master_bot_right)

        # --- Stop Condition ---
        # Check headset buttons to stop recording
        headset_data = headset.receive_data()
        if headset_data is not None:
            # Press 'A' (r_button_one) or 'X' (l_button_one) to stop
            if headset_data.r_button_one or headset_data.l_button_one:
                print("Episode finished by user.")
                break

        # --- Feedback ---
        if step_idx % 10 == 0:
            feedback.info = f"Recording Ep {episode_idx}: Step {step_idx}"
            headset.send_feedback(feedback)

        # --- Timing ---
        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step)) 

        step_idx += 1
    
    return True

def confirm_episode(headset: WebRTCHeadset, episode_idx):
    """Waits for user confirmation to save or discard the episode."""
    feedback = HeadsetFeedback()
    print("Waiting for user to confirm...")
    
    while True:
        start_time = time.time()
        headset_data = headset.receive_data()
        
        if headset_data is not None:
            # Left Hand 'X' button to Save/Next
            if headset_data.l_button_one == True: 
                return True
            # Left Hand 'Y' button to Discard/Redo
            elif headset_data.l_button_two == True: 
                return False       
               
        feedback.info = f"Ep {episode_idx} Done.\nPress 'X' to Save/Next.\nPress 'Y' to Discard/Redo."
        headset.send_feedback(feedback)
        
        time.sleep(0.02)

def main(cfg):
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

    current_episode = 0
    num_cameras = 2 # Right Wrist + Overhead
    
    # Create Dataset with Single-Arm Structure (7 DOF)
    dataset = LeRobotDataset.create(
        repo_id=cfg['repo_id'],
        root=os.path.join(cfg['root'], cfg['repo_id']),
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

    if dataset.num_episodes < cfg['num_episodes']:
        # Headset used for text feedback and button inputs only
        headset = WebRTCHeadset()
        headset.run_in_thread()

        # RealEnv (Right Arm Only)
        env = RealEnv(init_node=False) 

        while True:
            if dataset.num_episodes >= cfg['num_episodes']:
                break

            episode_idx = current_episode
            
            # Run Recording
            ok = run_episode(dataset, env, headset, master_bot_right, episode_idx, cfg['task'])

            if not ok:
                dataset.clear_episode_buffer()
                continue

            # Confirm Save/Discard
            ok = confirm_episode(headset, episode_idx)

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
                dataset.push_to_hub()
                print(f"Uploaded episodes {current_episode - cfg['batch_size']} to {current_episode - 1} to Hugging Face.")

    print("Data collection complete.")

if __name__ == "__main__":
    # ROS Setup
    parser = argparse.ArgumentParser(description="Record simulation episodes for AV Aloha (Leader-Follower Right Arm).")
    parser.add_argument("--num-episodes", type=int, default=80, help="Number of episodes to record.")
    parser.add_argument("--repo-id", type=str, default="Jinyu220/vedio_55", help="Repository ID for the dataset.")
    parser.add_argument("--root", type=str, default="vedio_55", help="Root directory for the dataset.")
    parser.add_argument("--task", type=str, default="vedio_55", help="Task name for the dataset.")
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
