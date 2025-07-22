# %%
from torch.utils.data import DataLoader, Subset
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset
from gaze_av_aloha.policies.gaze_policy.gaze_model import GazeModel
import torch
from torch import nn, Tensor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import cv2
from IPython.display import Video, display
import einops
import kornia.augmentation as K
import torch.nn.utils as nn_utils
import os
from hydra import initialize_config_dir, compose
from gaze_av_aloha.configs import Config
from torchvision.transforms import Resize, Normalize
import torch.nn.functional as F
from gaze_av_aloha.utils.dataset_utils import cycle
import argparse

# Define valid task choices
TASK_CHOICES = [
    "thread_needle",
    "pour_test_tube",
    "hook_package",
    "slot_insertion",
    "cube_transfer",
    "peg_insertion",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train gaze model for a specific AV-Aloha task")
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_CHOICES,
        required=True,
        help="Specify the AV-Aloha task to train on"
    )
    return parser.parse_args()

args = parse_args()
task = args.task

# %%
input_shape = (240, 320)
resize_shape=(120, 160)

dataset = f"iantc104/av_aloha_sim_{task}"
model_repo_id = f"iantc104/gaze_model_av_aloha_sim_{task}"
image_keys = [
    "observation.images.zed_cam_left",
]
eye_keys = [
    "left_eye",
]
batch_size = 64
num_steps = 30_000
lr = 1e-4
print(f"Training gaze model for {task} task")

# %%

delta_timestamps = {
    k: [0] for k in image_keys + eye_keys
}
dataset = AVAlohaDataset(
    repo_id=dataset,
    delta_timestamps=delta_timestamps,
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeModel(
    resize_shape=resize_shape,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

transforms = K.AugmentationSequential(
    K.RandomCrop(size=(int(input_shape[0]*0.9), int(input_shape[1]*0.9)), p=0.5),
    K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.1),
    K.RandomPerspective(distortion_scale=0.5, p=0.1),
    K.RandomHorizontalFlip(p=0.1),
    K.RandomRotation(degrees=15, p=0.1),
    K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.1),
    data_keys=["input", "keypoints"],
    same_on_batch=True,
)
resize = Resize(input_shape)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# %%
def denormalize_keypoints(
    keypoints: Tensor, image_size: tuple[int, int]
) -> Tensor:
    """
    Arguments:
        keypoints: Tensor of shape (..., 2)
            where the last dimension contains (x, y) coordinates of keypoints.
        image_size: Tuple (height, width) of the image.
    Denormalize keypoints from [-1, 1] range to pixel coordinates based on image size.
    """
    height, width = image_size
    return torch.stack(
        [
            ((keypoints[..., 0] + 1) / 2) * width,  # Denormalize x to pixel coordinates
            ((keypoints[..., 1] + 1) / 2) * height,  # Denormalize y to pixel coordinates
        ],
        dim=-1,
    )

def normalize_keypoints(
    keypoints: Tensor, image_size: tuple[int, int]
) -> Tensor:
    """
    Arguments:
        keypoints: Tensor of shape (..., 2)
            where the last dimension contains (x, y) coordinates of keypoints.
        image_size: Tuple (height, width) of the image.
    Normalize keypoints to [-1, 1] range based on image size.
    """
    height, width = image_size
    return torch.stack(
        [
            (keypoints[..., 0] / width) * 2 - 1,  # Normalize x to [-1, 1]
            (keypoints[..., 1] / height) * 2 - 1,  # Normalize y to [-1, 1]
        ],
        dim=-1,
    )


# %%

losses = []
dl_iter = cycle(dataloader)
for step in tqdm(range(num_steps)):
    model.train()
    batch = next(dl_iter)

    data = {k: v.to(device).squeeze(1) for k, v in batch.items() if k in image_keys + eye_keys}
    data["task"] = batch["task"]
    image = einops.rearrange(
        torch.stack([data[k] for k in image_keys], dim=1),
        'b n c h w -> (b n) c h w', n=len(image_keys)
    )
    eye = einops.rearrange(
        torch.stack([data[k] for k in eye_keys], dim=1),
        'b n e -> (b n) 1 e', n=len(image_keys)
    )
    image = resize(image)
    eye = denormalize_keypoints(eye, image.shape[-2:])
    image, eye = transforms(image, eye)
    eye = normalize_keypoints(eye, image.shape[-2:]).squeeze(1)
    image = normalize(image)
    
    optimizer.zero_grad()
    pred, _ = model(image)
    loss = F.mse_loss(pred, eye).mean()
    loss.backward()
    
    optimizer.step()
    losses.append(loss.item())

# %%
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"gaze_model_training_loss_{task}.png")

# %%
model.push_to_hub(model_repo_id)

