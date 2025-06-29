from torch import Tensor, nn
import torch

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

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)