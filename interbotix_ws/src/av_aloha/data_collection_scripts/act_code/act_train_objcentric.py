import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
# import matplotlib.pyplot as plt
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies import make_policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from pathlib import Path

try:
    from interbotix_xs_msgs.msg import JointSingleCommand
except ImportError:
    JointSingleCommand = None


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def kept_frame_indices(dataset, bad_episodes):
    bad_episodes = set(bad_episodes)
    epi = dataset.episode_data_index
    kept = []
    n_eps = len(epi["from"])
    for ep_idx in range(n_eps):
        start = int(epi["from"][ep_idx])
        end = int(epi["to"][ep_idx])
        if ep_idx in bad_episodes:
            continue
        kept.extend(range(start, end))
    return kept


def update_gripper(
    bot,
    trigger_pressed,
    close_position=-1.5,
    open_position=0.0,
):
    if JointSingleCommand is None:
        raise ImportError("interbotix_xs_msgs is required for gripper control.")
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = close_position if trigger_pressed else open_position
    bot.gripper.core.pub_single.publish(cmd)
    return cmd.cmd


REPLAY_MAX_JOINT_STEP = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)


def send_action(right_bot, action):
    if JointSingleCommand is None:
        raise ImportError("interbotix_xs_msgs is required for replaying actions.")
    action = action.detach().float().cpu().numpy().reshape(-1)
    if action.shape[0] < 7:
        raise ValueError(f"Expected action dim >= 7, got {action.shape[0]}")

    arm_cmd = action[:6]
    gripper_cmd = float(action[6])

    current_q = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)
    arm_cmd = np.clip(
        arm_cmd,
        current_q - REPLAY_MAX_JOINT_STEP,
        current_q + REPLAY_MAX_JOINT_STEP,
    )

    right_bot.arm.set_joint_positions(
        arm_cmd.tolist(),
        moving_time=0.14,
        accel_time=0.04,
        blocking=False,
    )

    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = gripper_cmd
    right_bot.gripper.core.pub_single.publish(cmd)

def extract_green_object_features(img):
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    moments = cv2.moments(mask)
    h, w = mask.shape

    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = w / 2
        cy = h / 2

    centroid = np.array([cx / w, cy / h], dtype=np.float32)
    return mask, centroid


def main():
    dataset_root_a = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433"
        #"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838_rgb_only"
    )
    # dataset_root_b = Path(
    #     "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_164157_rgb_only"
    # )

    # bad_episodes_a = {12}
    # bad_episodes_b = {6, 8, 9}

    output_directory = Path("outputs/act_grasp_cube_27_mask_centroid_1ex_10k_overfit")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="grasp_cube",
        root=dataset_root_a,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }

    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        # "observation.ee_pose",

        "observation.depth.right_wrist",
        "observation.depth.top_scene",
        "observation.depth_intrinsics.right_wrist",
        "observation.depth_intrinsics.top_scene",
        "observation.timestamps.right_wrist_depth",
        "observation.timestamps.top_scene_depth",

        # Optional later if you want to simplify to top camera only:
        # "observation.images.right_wrist",

        # Optional later if you want pure object-centric only:
        # "observation.images.top_scene",
    }

    input_features = {
        k: v
        for k, v in features.items()
        if k not in output_features and k not in excluded_features
    }

    input_features["observation.object_centroid"] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(2,),
    )

    print("Input features:", list(input_features.keys()))

    input_features["observation.object_mask"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 480, 640),
    )

    # If you want centroid-only later, comment out object_mask above
    # and also comment out the object_mask assignment in the train loop.

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=40,
        n_action_steps=10,
        use_vae=True,
        kl_weight=0.01,

        optimizer_lr=3e-4,
        optimizer_lr_backbone=1e-5,

        # n_obs_steps=3,
        # observation_delta_indices=[-2,-1,0]
    )

    print("Image features:", cfg.image_features)

    policy = make_policy(cfg, ds_meta=dataset_metadata)

    # CHANGED (lerobot v0.6.0): policies no longer carry normalization layers in
    # their weights. Normalization lives in an external processor pipeline built
    # from the dataset stats, and must be applied to every batch before
    # policy.forward(). See ACT_MODIFICATIONS.md.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=dataset_metadata.stats,
    )
    policy.train()
    policy.to(device)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    dataset_a = LeRobotDataset(
        repo_id="deviamar/transfer_flower",
        root=dataset_root_a,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    episode_indices = np.array(dataset_a.hf_dataset["episode_index"])

    target_episode = 0

    single_episode_frames = np.where(
        episode_indices == target_episode
    )[0]

    print("frames in episode:", len(single_episode_frames))

    train_dataset = Subset(
        dataset_a,
        single_episode_frames
    )
    # dataset_b = LeRobotDataset(
    #     repo_id="block_square",
    #     root=dataset_root_b,
    #     delta_timestamps=delta_timestamps,
    #     video_backend="pyav",
    # )

    # indices_a = kept_frame_indices(dataset_a, bad_episodes_a)
    # indices_b = kept_frame_indices(dataset_b, bad_episodes_b)

    # train_dataset = Subset(dataset_a, indices_a)

    # Restore this if you go back to two datasets:
    # train_dataset = ConcatDataset([
    #     Subset(dataset_a, indices_a),
    #     Subset(dataset_b, indices_b),
    # ])

    # print(f"dataset_a kept frames: {len(indices_a)}")
    # print(f"dataset_b kept frames: {len(indices_b)}")
    # print(f"total kept frames: {len(train_dataset)}")

    batch_size = 8
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    training_steps = 10000
    log_freq = 100

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device, non_blocking=True)
                if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            for key in cfg.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key][:, [2, 1, 0], :, :]
                elif batch[key].ndim == 5:
                    batch[key] = batch[key][:, :, [2, 1, 0], :, :]

            top_imgs = batch["observation.images.top_scene"]

            if top_imgs.ndim == 5:
                top_imgs_now = top_imgs[:, -1]
            elif top_imgs.ndim == 4:
                top_imgs_now = top_imgs
            else:
                raise ValueError(f"Unexpected top_scene shape: {top_imgs.shape}")

            masks = []
            centroids = []

            for img in top_imgs_now:
                img_np = img.permute(1, 2, 0).detach().cpu().numpy()

                mask, centroid = extract_green_object_features(img_np)

                mask = mask.astype(np.float32) / 255.0
                mask_rgb = np.repeat(mask[None], 3, axis=0)

                masks.append(mask_rgb)
                centroids.append(centroid.astype(np.float32))

            masks = torch.from_numpy(np.stack(masks)).to(device)
            centroids = torch.from_numpy(np.stack(centroids)).to(device)

            batch["observation.object_mask"] = masks
            batch["observation.object_centroid"] = centroids

            # CHANGED (lerobot v0.6.0): images arrive as uint8 and the normalizer
            # expects float in [0, 1], then the preprocessor applies the mean/std
            # normalization that used to live inside the policy.
            for cam_key in cfg.image_features:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

            batch = preprocessor(batch)

            loss, _ = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")

            if step == 0:
                print(batch["observation.object_centroid"].shape)

            step += 1
            if step % 1000 == 0 and step > 0:
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                    },
                    output_directory / f"checkpoint_{step}.pt",
                )
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)

    # CHANGED (lerobot v0.6.0): normalization stats now live in the processor
    # pipelines rather than the model weights, so they must be saved next to the
    # policy. Without these the checkpoint cannot be run at rollout time.
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    ckpt_path = output_directory / "checkpoint.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        ckpt_path,
    )


if __name__ == "__main__":
    main()
