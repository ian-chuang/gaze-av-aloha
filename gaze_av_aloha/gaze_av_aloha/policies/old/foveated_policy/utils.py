from torch import Tensor, nn
import torchvision
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

def crop_with_keypoints(images: Tensor, crop_shape: tuple, keypoints: Tensor = None, crop_is_random: bool =False):
    image_shape = images.shape[-2:]
    if crop_shape != image_shape:
        if crop_is_random:
            crop_indices = torchvision.transforms.RandomCrop.get_params(
                images, output_size=crop_shape
            )
            i, j, h, w = crop_indices  
        else:
            i = (image_shape[0] - crop_shape[0]) // 2
            j = (image_shape[1] - crop_shape[1]) // 2
            h = crop_shape[0]
            w = crop_shape[1]
        images = torchvision.transforms.functional.crop(images, i, j, h, w)
    else:
        i, j, h, w = 0, 0, image_shape[0], image_shape[1]

    if keypoints is not None:
        keypoints = torch.stack([
            (((keypoints[..., 0]+1)/2) * image_shape[1] - j) / w * 2 - 1,
            (((keypoints[..., 1]+1)/2) * image_shape[0] - i) / h * 2 - 1,
        ], dim=-1)

    return images, keypoints

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)