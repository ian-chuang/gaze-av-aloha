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
import time
import cv2

from gaze_av_aloha.policies.gaze_policy.gaze_policy import GazePolicy


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation."""
    return_observations = {}
    imgs = {f"observation.images.{key}": img for key, img in observations["images"].items()}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img.copy()).unsqueeze(0)

        # sanity check channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but got {img.shape=}"
        assert img.dtype == torch.uint8, f"expect torch.uint8, but got {img.dtype=}"

        # convert to channel first float32 [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32) / 255
        return_observations[imgkey] = img

    return_observations["observation.state"] = torch.from_numpy(
        observations["joints"]
    ).float().unsqueeze(0)
    return return_observations


def load_policy(policy_path: Path, device):
    """Load a policy and apply EMA weights."""
    policy = GazePolicy.from_pretrained(policy_path / "policy")
    ema = policy.get_ema()
    training_state = torch.load(policy_path / "training_state.pt")
    ema.load_state_dict(training_state["ema"])
    for s_param, param in zip(ema.shadow_params, policy.parameters()):
        if s_param.shape == param.shape:
            param.data.copy_(s_param.to(param.device).data)
        else:
            print(f"Skipping param mismatch: EMA {s_param.shape} vs Model {param.shape}")
    return policy.to(device)


def eval(args):
    policy_paths = [Path(p) for p in args['policies']]   # 多个policy路径
    episode_len = args['episode_len']
    num_episodes = args['num_episodes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预加载所有 policies
    policies = []
    for p in policy_paths:
        print(f"Loading policy from {p}")
        policies.append(load_policy(p, device))

    n_interpolation_steps = round(FPS / policies[0].task_cfg.fps)
    print(f"Using {n_interpolation_steps} interpolation steps for actions.")

    # setup the environment
    env = RealEnv(init_node=True, stereo_cam_idx=18)

    success_episodes = []
    failed_episodes = []

    for i in range(num_episodes):
        # 按顺序轮流选policy
        policy_idx = i % len(policies)
        policy = policies[policy_idx]
        print(f"\n=== Running episode {i+1}/{num_episodes} with policy {policy_idx+1} ===")

        # reset
        policy.reset()
        observation, info = env.reset()
        ctrl = observation['control']

        input("Press Enter to start...")

        viz_videos = {}
        raw_videos = {}
        for _ in tqdm(range(episode_len)):
            observation = preprocess_observation(observation)
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

            with torch.inference_mode():
                action, viz = policy.select_action(observation, return_viz=True)

            # 保存可视化
            for key, images in viz.items():
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                images = images.cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = (images.numpy() * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)
                viz_videos.setdefault(key, []).append(images)

            raw_image = observation["observation.images.left_eye_cam"].cpu().numpy()
            raw_image = einops.rearrange(raw_image, "b c h w -> h (b w) c")
            raw_image = (raw_image * 255).astype(np.uint8)
            raw_videos.setdefault("raw", []).append(raw_image)
            # 执行动作
            action = action.to("cpu").numpy()
            actions = np.linspace(ctrl, action[0], n_interpolation_steps+1)[1:]
            for k in range(n_interpolation_steps):
                observation, info = env.step_action(actions[k])
                time.sleep(REAL_DT)
                viz_image = observation["images"]["left_eye_cam"]
                viz_videos.setdefault("eval", []).append(viz_image)

            ctrl = observation['control']

        # 保存视频
        video_dir = policy_paths[policy_idx] / "eval_distractors" / f"rollout_distractor_jinyus_{i+2}"
        os.makedirs(str(video_dir), exist_ok=True)
        for key, video in viz_videos.items():
            video_path = video_dir / f"{key}.mp4"
            speed_factor = len(video) / (episode_len * n_interpolation_steps)
            imageio.mimsave(str(video_path), video, fps=round(FPS * speed_factor))

        print(f"Videos saved to {video_dir}")
        raw_video_path = video_dir / f"raw_{i}.mp4"
        imageio.mimsave(str(raw_video_path), raw_videos["raw"], fps=round(FPS * speed_factor/3.0))
        while True:
            is_success = input("Success? (y/n): ").strip().lower()
            if is_success in ('y', 'n'):
                break
            print("Please enter 'y' or 'n'.")
        if is_success == 'y':
            success_episodes.append((i+1, policy_idx+1))
        else:
            failed_episodes.append((i+1, policy_idx+1))

        print('Success episodes:', success_episodes)
        print('Failed episodes:', failed_episodes)
        input("Press Enter to reset env...")

'''
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-17_16-25-02_put_coin_resnet
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-18_23-27-40_fine-hang_coin_fine_v4
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-14_16-31-53_fov-unet-coinv2
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-17_17-36-04_put_coin_dino


"/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-19_17-30-56_hang_ring_resnet/checkpoints/0000030000",
            "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-18_12-38-39_fine-hang_cirle_v4/checkpoints/0000030000",
            "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-14_17-58-17_fov-unet-hang_ring/checkpoints/0000030000",

            
hang_ring:
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-22_23-04-38_fov-unet-augementation_hang_ring/checkpoints/0000030000
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-23_00-00-55_fine-augementation_hang_ring/checkpoints/0000030000
/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-19_17-30-56_hang_ring_resnet/checkpoints/0000030000

'''
def main():
    import argparse, traceback, rospy
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--policies', nargs='+',
        default=[
            #"/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-24_02-12-39_fov-unet-augementation_hangv4_ring/checkpoints/0000030000",
            #  "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-04_16-24-19_fov-unet-augementation_hang__Ian/checkpoints/0000030000",
            # "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-04_17-22-32_hang_ring_resnet/checkpoints/0000030000",
            # "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-16_19-30-35_hang_ring_dino/checkpoints/0000030000",
            # "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-04_20-13-21_fov-unet-augementation_shoot_Ian/checkpoints/0000030000",
            # "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-05_14-06-06_fine-augementation_shoot/checkpoints/0000030000"
            "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-05_18-15-35_fine-augementation_toothbrush/checkpoints/0000030000",
            "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-05_23-34-13_fov-unet-augementation_toothbrush_Ian/checkpoints/0000030000",
    #    "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-09_16-46-32_fov-unet-augementation_hook_Ian/checkpoints/0000030000",
        
           #Z "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-08-19_17-30-56_hang_ring_resnet/checkpoints/0000030000",
          # "/home/jinyu/GitHub/gaze-av-aloha/outputs/2025-09-04_17-22-32_hang_ring_resnet/checkpoints/0000030000",
     
            
        ],
        help='List of policy checkpoint dirs, will be cycled episode by episode'
    )
    parser.add_argument('--episode_len', type=int, default=150)
    parser.add_argument('--num_episodes', type=int, default=100)
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
