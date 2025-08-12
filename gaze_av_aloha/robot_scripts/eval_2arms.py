from pathlib import Path
from gaze_av_aloha.robot.env_no_left import RealEnv
from gaze_av_aloha.robot.config import REAL_DT, FPS
from tqdm import tqdm
import einops
import numpy as np
import torch
import imageio
import os
from torch import Tensor
import torchvision.transforms as v2
import time
import safetensors.torch
from diffusers.training_utils import EMAModel
from gaze_av_aloha.policies.gaze_policy.gaze_policy import GazePolicy
import cv2

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    return_observations = {}
    imgs = {f"observation.images.{key}": img for key, img in observations["images"].items()}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img.copy()).unsqueeze(0)

        # sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

        # sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255

        return_observations[imgkey] = img


    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["joints"]).float().unsqueeze(0)
    return return_observations

def eval(args):
    policy_path = Path(args['policy'])
    episode_len = args['episode_len']
    num_episodes = args['num_episodes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = GazePolicy.from_pretrained(policy_path / "policy")
    ema = policy.get_ema()
    training_state = torch.load(policy_path / "training_state.pt")
    ema.load_state_dict(training_state["ema"])
    for s_param, param in zip(ema.shadow_params, policy.parameters()):
        if s_param.shape == param.shape:
            param.data.copy_(s_param.to(param.device).data)
        else:
            print(f"Skipping param with shape mismatch: EMA {s_param.shape} vs Model {param.shape}")
    policy = policy.to(device)

    n_interpolation_steps = round(FPS / policy.task_cfg.fps)
    print(f"Using {n_interpolation_steps} interpolation steps for actions.")

    # setup the environment
    env = RealEnv(init_node=True, stereo_cam_idx=24)

    success_episodes = []
    failed_episodes = []
    # run the policy for the specified number of steps
    for i in range(num_episodes):
        print("Resetting environment...")
        # reset the environment
        policy.reset()
        observation, info = env.reset()
        ctrl = observation['control']

        print(f"Running episode {i+1}/{num_episodes}")
        input("Press Enter to start...")

        viz_videos = {}
        for _ in tqdm(range(episode_len)):
            observation = preprocess_observation(observation)
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

            with torch.inference_mode():
                action, viz = policy.select_action(observation, return_viz=True)

            for key, images in viz.items():
                # get images
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = images.cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = (images.numpy() * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)
                if key in viz_videos:
                    viz_videos[key].append(images)
                else:
                    viz_videos[key] = [images]

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

            actions = np.linspace(ctrl, action[0], n_interpolation_steps+1)[1:]
            for k in range(n_interpolation_steps):
                observation, info = env.step_action(actions[k])
                time.sleep(REAL_DT)

                viz_image = observation["images"]["left_eye_cam"]
                key = "eval"
                if key in viz_videos:
                    viz_videos[key].append(viz_image)
                else:
                    viz_videos[key] = [viz_image]

            ctrl = observation['control']

        print("Episode finished, saving frames as video...")
        # Encode all frames into a mp4 video.


        video_dir = policy_path / "eval_distractors" / f"rollout_{i+35}"
        os.makedirs(str(video_dir), exist_ok=True)
        for key, video in viz_videos.items():
            video_path = video_dir / f"{key}.mp4"
            speed_factor = len(video) / (episode_len * n_interpolation_steps)
            imageio.mimsave(str(video_path), video, fps=round(FPS * speed_factor))


        print(f"Videos saved to {video_dir}")
        while True:
            is_success = input("Success? (y/n): ").strip().lower()
            if is_success in ('y', 'n'):
                break
            print("Please enter 'y' for success or 'n' for failure.")
        if is_success == 'y':
            success_episodes.append(i+1)
        else:
            failed_episodes.append(i+1)
        print('Success episodes:', success_episodes)
        print('Failed episodes:', failed_episodes)
        input("Press Enter to reset env...")


def main():

    """
    
    python eval.py \
        --policy /home/jinyu/GitHub/gaze-av-aloha/outputs/2025-07-27_23-18-17_foveated_put_tube1_train_ZJY/checkpoints/0000030000 \
        --episode_len 300 \
        --num_episodes 10

    """

    import argparse
    import traceback
    import rospy
    # add arg for policy
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default="/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-06_22-22-45_fine-cube_v2/checkpoints/0000030000", help='Path to the policy checkpoint directory parent folder of policy/model.safetensors')
    parser.add_argument('--episode_len', type=int, default=125, help='Length of the episode')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to run')
    # convert to dict
    args = vars(parser.parse_args())

    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        eval(args)
    except Exception as e:
        print(f"An error occured: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down...")
        os._exit(42)

if __name__ == "__main__":
    main()
