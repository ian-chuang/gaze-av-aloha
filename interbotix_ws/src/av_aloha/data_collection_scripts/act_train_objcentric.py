from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.lerobot_dataset import (LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.factory import make_policy
# from lerobot.common.policies.factory import make_pre_post_processors
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature
from torch.utils.data import Subset

import cv2
import numpy as np

def extract_green_object_features(img):
    """
    img: numpy RGB image in [0,1] or [0,255]
    returns:
        mask: (H,W) float32
        centroid: (2,)
    """

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)

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

    centroid = np.array([
        cx / w,
        cy / h,
    ], dtype=np.float32)

    mask = mask.astype(np.float32) / 255.0

    return mask, centroid

def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]

def main():
    dataset_root = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260524_224911"
    )
    output_directory = Path("outputs/act_rgb_mask_centroid_ee_vae_c10_a5")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="block_square",
        root=dataset_root,
    )
    # dataset_metadata = LeRobotDatasetMetadata(dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    # input_features = {k: v for k, v in features.items() if k not in output_features}
    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        # "observation.ee_pose",

        # "observation.images.right_wrist",
        # "observation.images.top_scene",
    }

    input_features = {
        k: v
        for k, v in features.items()
        if (
            k not in output_features
            and k not in excluded_features
        )
    }

    input_features["observation.object_centroid"] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(2,),
    )

    input_features["observation.object_mask"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 480, 640),
    )

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=5,
        use_vae=True,
        kl_weight=1.0,

        optimizer_lr=3e-4,
        optimizer_lr_backbone=1e-5,

        # n_obs_steps=3,
        # observation_delta_indices=[-2,-1,0]
    )

    # print("Config")
    print(cfg)
    # policy = ACTPolicy(cfg)
    # preprocessor, postprocessor = make_pre_post_processors(
    #     cfg, dataset_stats=dataset_metadata.stats
    # )
    # print("action delta indices")
    print(cfg.action_delta_indices)
    # print("observation delta indices")
    print(cfg.observation_delta_indices)
    # print("chunk size")
    print(cfg.chunk_size)
    # print("action steps")
    print(cfg.n_action_steps)

    policy = make_policy(cfg, ds_meta=dataset_metadata)

    

    policy.train()
    policy.to(device)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # dataset = LeRobotDataset(dataset_root, delta_timestamps=delta_timestamps)
    # dataset = LeRobotDataset(dataset_root, delta_timestamps=delta_timestamps, video_backend="pyav")

    dataset = LeRobotDataset(
        repo_id="block_square",
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    # episode_indices = [
    #     i
    #     for i in range(len(dataset))
    #     if dataset.hf_dataset[i]["episode_index"] == 0
    # ]

    # dataset = Subset(dataset, episode_indices)

    # sample = dataset[0]

    # for k, v in sample.items():
    #     if torch.is_tensor(v):
    #         print(k, v.shape)
    #     else:
    #         print(k, type(v))

    batch_size = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    training_steps = 2000
    log_freq = 10

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
                batch[key] = batch[key][:, [2,1,0], :, :]

            top_imgs = batch["observation.images.top_scene"]

            masks = []
            centroids = []

            for img in top_imgs:

                img_np = (
                    img.permute(1,2,0)
                    .cpu()
                    .numpy()
                )

                mask, centroid = extract_green_object_features(img_np)

                if step == 0:
                    plt.subplot(1,2,1)
                    plt.imshow(img_np)

                    plt.subplot(1,2,2)
                    plt.imshow(mask)

                    plt.show()

                mask_rgb = np.repeat(mask[None], 3, axis=0)
                masks.append(mask_rgb)
                centroids.append(centroid)
            
            masks = torch.tensor(
                np.stack(masks),
                device=device,
            )

            centroids = torch.tensor(
                np.stack(centroids),
                device=device,
            )

            batch["observation.object_mask"] = masks
            batch["observation.object_centroid"] = centroids

            # for i in range(20):
            #     img = batch["observation.images.top_scene"][i]

            #     print(img.min(), img.max(), img.dtype)

            #     img = img.permute(1,2,0).cpu().numpy()

            #     img = np.clip(img, 0, 1)

            #     plt.imshow(img)
            #     plt.show()

            loss, _ = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)
    # preprocessor.save_pretrained(output_directory)
    # postprocessor.save_pretrained(output_directory)
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




