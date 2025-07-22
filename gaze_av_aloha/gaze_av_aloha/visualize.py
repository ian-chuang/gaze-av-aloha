from pathlib import Path
import einops
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tqdm import trange
from gaze_av_aloha.utils.policy_utils import get_device_from_parameters, preprocess_observation
from gaze_av_aloha.utils.utils import (
    inside_slurm,
)
from gaze_av_aloha.policies.policy import Policy
import cv2
import imageio
import os

def visualize_policy(
    env: gym.vector.VectorEnv,
    policy: Policy,
    videos_dir: Path,
    options: dict | None = None,
    seed: int | None = None,
    steps: int | None = None,
) -> dict:
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    policy.eval()

    observation, info = env.reset(seed=seed, options=options)

    step = 0
    # Keep track of which environments are done.
    progbar = trange(
        steps,
        desc=f"Running rollout with {steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )

    viz_videos = {}
    max_height = 240

    if "prompt" in options:
        task = [options["prompt"]] * env.num_envs
    else:
        task = None
    
    # features_video = []
    while step < steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)

        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }
        if task is not None:
            observation["task"] = task

        with torch.inference_mode():
            action, viz = policy.select_action(observation, return_viz=True)

        # Convert to CPU / numpy.
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)

        step += 1
        progbar.update()
        if isinstance(viz, dict):
            for key, images in viz.items():
                # get images
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = images.cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = (images.numpy() * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)
                if images.shape[0] > max_height:
                    scale = max_height / images.shape[0]
                    new_width = int(images.shape[1] * scale)
                    images = cv2.resize(images, (new_width, max_height), interpolation=cv2.INTER_AREA)
                if key in viz_videos:
                    viz_videos[key].append(images)
                else:
                    viz_videos[key] = [images]

    video_paths = []    
    os.makedirs(str(videos_dir), exist_ok=True)
    for key, video in viz_videos.items():
        video_path = videos_dir / f"{key}.mp4"
        video_paths.append(str(video_path))
        speed_factor = len(video) / steps
        imageio.mimsave(str(video_path), video, fps=env.unwrapped.metadata["render_fps"] * speed_factor)

    return video_paths


class NoTerminationWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, False, info  # Always return False for termination/truncation
        