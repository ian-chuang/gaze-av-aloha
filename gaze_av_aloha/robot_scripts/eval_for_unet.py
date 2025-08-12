import argparse
import os
import torch
import gymnasium as gym
from gaze_av_aloha.policies.foveated_vit_policy import FoveatedViTPolicy  # 确保这是你的策略类
from gaze_av_aloha.scripts.eval_policy import eval_policy  # 确保这是你的评估函数
from pathlib import Path
from tqdm import tqdm
import einops
import numpy as np
import cv2
import imageio

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
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

    # Add state observation
    return_observations["observation.state"] = torch.from_numpy(observations["joints"]).float().unsqueeze(0)
    return return_observations

def eval(args):
    policy_path = Path(args['policy'])
    episode_len = args['episode_len']
    num_episodes = args['num_episodes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用训练时的设备

    # 加载策略模型
    policy = FoveatedViTPolicy(
        use_gaze_as_action=False,
        gaze_model_repo_id="Jinyu220/gaze_model_av_aloha_real_insert_straw",
        vision_encoder_kwargs={"repo_id": "iantc104/mae_vitb_foveated_vit"},
        optimizer_lr_backbone=1e-5,
    )

    # 加载权重
    checkpoint = torch.load(policy_path / "model.pth", map_location=device)
    policy.load_state_dict(checkpoint)
    policy = policy.to(device)
    policy.eval()

    # 创建环境
    env = gym.make("insert_straw-v0")  # 替换为你的环境名称

    success_episodes = []
    failed_episodes = []

    for i in range(num_episodes):
        print(f"Running episode {i+1}/{num_episodes}")
        policy.reset()
        observation, info = env.reset()

        viz_videos = {}
        for _ in tqdm(range(episode_len)):
            observation = preprocess_observation(observation)
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

            with torch.inference_mode():
                action = policy.select_action(observation)

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

            observation, reward, terminated, truncated, info = env.step(action[0])

            # Save visualization frames
            viz_image = observation["images"]["left_eye_cam"]
            key = "eval"
            if key in viz_videos:
                viz_videos[key].append(viz_image)
            else:
                viz_videos[key] = [viz_image]

            if terminated or truncated:
                break

        print("Episode finished, saving frames as video...")
        video_dir = policy_path / "eval" / f"rollout_{i+1}"
        os.makedirs(str(video_dir), exist_ok=True)
        for key, video in viz_videos.items():
            video_path = video_dir / f"{key}.mp4"
            imageio.mimsave(str(video_path), video, fps=30)

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

def main():
    """
    python eval.py \
        --policy /path/to/your/checkpoint \
        --episode_len 300 \
        --num_episodes 10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str,default="/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-04_15-19-29_fov-unet-straw2/checkpoints/0000030000", help='Path to the policy checkpoint directory')
    parser.add_argument('--episode_len', type=int,default=300, help='Length of the episode')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to run')
    args = vars(parser.parse_args())

    eval(args)

if __name__ == "__main__":
    main()