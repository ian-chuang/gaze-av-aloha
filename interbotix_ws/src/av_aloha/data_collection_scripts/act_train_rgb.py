from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.lerobot_dataset import (LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.act.configuration_act import ACTConfig
from interbotix_xs_msgs.msg import JointSingleCommand
import torchvision.transforms as T

resize = T.Resize((224,224))

import numpy as np

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
        kept.extend(range(start, end, 4))
    return kept

def update_gripper(
    bot,
    trigger_pressed,
    close_position=-1.5,
    open_position=0.0,
):
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = close_position if trigger_pressed else open_position
    bot.gripper.core.pub_single.publish(cmd)
    return cmd.cmd

REPLAY_MAX_JOINT_STEP = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)

def send_action(right_bot, action):
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

def main():
    # dataset_root_a = Path(
    #     "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838_rgb_only"
    # )

    # dataset_root_b = Path(
    #     "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_164157_rgb_only"
    # )

    # dataset_root_c = Path(
    #     "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp cube/20260528_235229"
    # )

    # bad_episodes_a = {12}
    # bad_episodes_b = {6, 8, 9}      # if these are still the bad ones
    bad_episodes = {3}

    dataset_root = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp cube/20260528_235229"
        #"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube_downsampled_20"
    )

    output_directory = Path("outputs/act_pickup_block_square_20_12.5Hz")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="grasp_cube",
        root=dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        "observation.ee_pose",

        "observation.depth.right_wrist",
        "observation.depth.top_scene",
        "observation.depth_intrinsics.right_wrist",
        "observation.depth_intrinsics.top_scene",
        "observation.timestamps.right_wrist_depth",
        "observation.timestamps.top_scene_depth",

        # "observation.images.right_wrist",
        # "observation.images.top_scene",
    }
    input_features = {
        k: v
        for k, v in features.items()
        if k not in output_features and k not in excluded_features
    }


    print("INPUT FEATURES:")
    for k, v in input_features.items():
        print(k, v.type, v.shape)

    print("\nOUTPUT FEATURES:")
    for k, v in output_features.items():
        print(k, v.type, v.shape)

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=20,
        n_action_steps=5,
        use_vae=True,
        kl_weight=1.0,
        optimizer_lr=3e-4,
        optimizer_lr_backbone=1e-5,
    )

    # DIDN'T WORK
    # cfg = ACTConfig(
    #     input_features=input_features,
    #     output_features=output_features,
    #     chunk_size=20,
    #     n_action_steps=5,
    #     action_delta_indices=[
    #         0, 4, 8, 12, 16,
    #         20, 24, 28, 32, 36,
    #         40, 44, 48, 52, 56,
    #         60, 64, 68, 72, 76,
    #     ],
    #     use_vae=True,
    #     kl_weight=1.0,
    #     optimizer_lr=3e-4,
    #     optimizer_lr_backbone=1e-5,
    # )

    policy = make_policy(cfg, ds_meta=dataset_metadata)
    policy.train()
    policy.to(device)

    print("policy has been made")

    # DIDN'T WORK
    # cfg.action_delta_indices = [
    #     0,  4,  8, 12, 16,
    #     20, 24, 28, 32, 36,
    #     40, 44, 48, 52, 56,
    #     60, 64, 68, 72, 76,
    # ]

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    print("creating datasets")

    print("chunk_size:", cfg.chunk_size)
    print("n_action_steps:", cfg.n_action_steps)
    print("action_delta_indices:", cfg.action_delta_indices)
    print("observation_delta_indices:", cfg.observation_delta_indices)

    print("fps =", dataset_metadata.fps)

    print("loading dataset")
    dataset_a = LeRobotDataset(
        repo_id="grasp_cube",
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    print("loaded dataset")

    

    # dataset_b = LeRobotDataset(
    #     repo_id="block_square",
    #     root=dataset_root_b,
    #     delta_timestamps=delta_timestamps,
    #     video_backend="pyav",
    # )
    # print("loaded dataset B")

    # dataset_c = LeRobotDataset(
    #     repo_id="grasp_cube",
    #     root=dataset_root_c,
    #     delta_timestamps=delta_timestamps,
    #     video_backend="pyav",
    # )

    indices_a = kept_frame_indices(dataset_a, bad_episodes)
    # indices_b = kept_frame_indices(dataset_b, bad_episodes_b)
    # indices_c = kept_frame_indices(dataset_c, bad_episodes_c)

    print(f"dataset_a kept frames: {len(indices_a)}")

    train_dataset = ConcatDataset([
        Subset(dataset_a, indices_a),
        # Subset(dataset_b, indices_b),
        #Subset(dataset_c, indices_c),
    ])

    # print(f"dataset_b kept frames: {len(indices_b)}")
    #print(f"dataset_c kept frames: {len(indices_c)}")
    print(f"total kept frames: {len(train_dataset)}")
    print("Episodes A:", len(dataset_a.episode_data_index["from"]))
    # print("Episodes B:", len(dataset_b.episode_data_index["from"]))
    #print("Episodes C:", len(dataset_c.episode_data_index["from"]))

    print("Dataset size:", len(train_dataset))

    print("creating dataloader")

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
    print("starting training loop")
    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            if step == 0:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        print(k, v.shape)

            for key in cfg.image_features:
                batch[key] = batch[key][:, [2, 1, 0], :, :]

                if batch[key].ndim == 4:
                    batch[key] = resize(batch[key])

                elif batch[key].ndim == 5:
                    B, T, C, H, W = batch[key].shape
                    x = batch[key].reshape(B * T, C, H, W)
                    x = resize(x)
                    batch[key] = x.reshape(B, T, C, 224, 224)

            loss, _ = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")

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
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        output_directory / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()