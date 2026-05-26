from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.lerobot_dataset import (LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.factory import make_policy
# from lerobot.common.policies.factory import make_pre_post_processors
from lerobot.common.policies.act.configuration_act import ACTConfig

import numpy as np

def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]

def main():
    dataset_root = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260524_224911"
    )
    output_directory = Path("outputs/act_all_rgb_with_vae_c10_a5")
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
        "observation.ee_pose",

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




