from torch import Tensor, nn
import torch
from torchvision.ops import roi_align

def apply_rotary_embed(x, thetas):
    assert x.shape[-2:] == thetas.shape[-2:]
    x1, x2 = x.chunk(2, dim=-1)
    x_rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * thetas.cos() + x_rotate_half * thetas.sin()

def get_1d_rotary_embed(dim:int, pos: Tensor, base:int=10000):
    assert dim % 2 == 0
    thetas = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    thetas = pos.unsqueeze(-1) * thetas  # (N, D/2)
    thetas = torch.cat((thetas, thetas), dim=-1)  # (N, D)
    return thetas    

def get_foveated_pos_embed(
    centers: Tensor,
    grid_shape: tuple,
    crop_scale: float,
    feat_shape: tuple,
    dim: int = 384,
    base: int = 10000,
):
    batch_size = centers.shape[0]
    assert centers.shape[-1] == 2, "centers should have shape (B, ..., 2) where last dimension is (x, y) coordinates"

    crop_shape = (crop_scale * grid_shape[0], crop_scale * grid_shape[1])
    patch_shape = (crop_shape[0] / feat_shape[0], crop_shape[1] / feat_shape[1])
    num_axis = len(centers.shape[1:-1]) + 2
    assert dim % num_axis == 0
    axis_dim = dim // num_axis
    assert axis_dim % 2 == 0

    # Create meshgrid along eash axis
    axis_grids = list(torch.meshgrid(
        *[torch.arange(d, dtype=torch.float32, device=centers.device) for d in centers.shape[1:-1]],
        torch.linspace(-crop_shape[0]/2 + patch_shape[0]/2, crop_shape[0]/2 - patch_shape[0]/2, feat_shape[0], device=centers.device),
        torch.linspace(-crop_shape[1]/2 + patch_shape[1]/2, crop_shape[1]/2 - patch_shape[1]/2, feat_shape[1], device=centers.device),
        indexing="ij",
    ))
    for i in range(len(axis_grids) - 2):
        axis_grids[i] = axis_grids[i].unsqueeze(0).expand(batch_size, *axis_grids[i].shape)
    axis_grids[-2] = axis_grids[-2].unsqueeze(0) + centers[...,1].unsqueeze(-1).unsqueeze(-1) * grid_shape[0] / 2
    axis_grids[-1] = axis_grids[-1].unsqueeze(0) + centers[...,0].unsqueeze(-1).unsqueeze(-1) * grid_shape[1] / 2
    # Compute position embeddings for each axis and concatenate
    axis_thetas = [
        get_1d_rotary_embed(axis_dim, axis_grid.flatten(start_dim=1), base)
        for axis_grid in axis_grids
    ]
    thetas = torch.cat(axis_thetas, dim=-1)

    return thetas

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

def batched_crop(images: torch.Tensor, crop_bounds: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """
    Crops images based on the provided crop bounds and resizes them to the specified output shape.
    Args:
        images (torch.Tensor): Input images of shape [B, C, H, W].
        crop_bounds (torch.Tensor): Crop bounds of shape [B, 2, 2], where each entry is [[x1, y1], [x2, y2]].
        out_shape (tuple): Desired output shape for the cropped images, e.g., (H_out, W_out).
    Returns:
        torch.Tensor: Cropped and resized images of shape [B, C, H_out, W_out].
    """
    assert images.ndim == 4, "images should have shape [B, C, H, W]"
    assert crop_bounds.ndim == 3 and crop_bounds.shape[1:] == (2, 2), "crop_bounds should have shape [B, 2, 2] where each entry is [[x1, y1], [x2, y2]]"
    assert len(out_shape) == 2, "out_shape should be a tuple of (height, width)"
    assert images.shape[0] == crop_bounds.shape[0], "Batch size of images and crop_bounds must match"

    B = crop_bounds.shape[0]
    batch_indices = torch.arange(B, device=crop_bounds.device, dtype=crop_bounds.dtype).unsqueeze(1)

    # Concatenate batch index with crop coordinates
    boxes = torch.cat([
        batch_indices,                      # (B, 1)
        crop_bounds[:, 0, 0:1],            # x1
        crop_bounds[:, 0, 1:2],            # y1
        crop_bounds[:, 1, 0:1],            # x2
        crop_bounds[:, 1, 1:2],            # y2
    ], dim=1)  # (B, 5)

    return roi_align(images, boxes, output_size=out_shape, aligned=True)

def get_crop_bounds(center: torch.Tensor, crop_shape: tuple) -> torch.Tensor:
    """
    Arguments:
        center: Tensor of shape (..., 2) where the last dimension contains (x, y) coordinates of the center of the crop.
        crop_size: Tensor of shape (2,)
            Size of the crop in the format (height, width).
    Returns:
        Tensor of shape (..., 2, 2) where the last dimension contains the lower and upper corners of the crop.
        The shape of the output is determined by the shape of the `center` tensor.
        The output will be in the format [[[x1, y1], [x2, y2]]], where (x1, y1) is the lower corner
        and (x2, y2) is the upper corner of the crop.
    """
    assert center.ndim >= 2 and center.shape[-1] == 2, "center should have shape (..., 2) where last dimension is (x, y) coordinates"
    assert len(crop_shape) == 2, "crop_shape should be a tuple of (height, width)"

    crop_size = torch.tensor(crop_shape[::-1], device=center.device, dtype=center.dtype)
    lower_corner = (center - crop_size / 2)
    upper_corner = lower_corner + crop_size
    return torch.stack([lower_corner, upper_corner], dim=1)  

def crop_at_kp(images: Tensor, crop_scale: float, kp: Tensor, out_shape: tuple = None) -> Tensor:
    assert images.ndim == 4, "images should have shape [B, C, H, W]"
    assert kp.ndim == 2 and kp.shape[-1] == 2, "kp should have shape [B, 2] where B is the batch size"
    assert kp.shape[0] == images.shape[0], "kp batch size must match images batch size"

    image_shape = images.shape[-2:]
    crop_shape = (crop_scale * image_shape[0], crop_scale * image_shape[1])
    center = denormalize_keypoints(kp, image_size=image_shape)
    if out_shape is None:
        out_shape = crop_shape
    crop_bounds = get_crop_bounds(
        center=center,
        crop_shape=crop_shape,
    ) 
    return batched_crop(
        images=images,
        crop_bounds=crop_bounds,
        out_shape=out_shape,
    )

def random_crop(images: Tensor, crop_scale: float, kp: Tensor = None, out_shape: tuple = None, random: bool = True) -> tuple[Tensor, Tensor]:
    """
    Args:
        images (Tensor): Input images of shape [B, C, H, W].
        crop_scale (float): Scale factor for the crop size relative to the image size.
        kp (Tensor, optional): Keypoints of shape [B, N, 2] where N is the number of keypoints. they are transformed to the new crop bounds.
        out_shape (tuple, optional): Desired output shape for the cropped images, e.g., (H_out, W_out).
        random (bool): If True, crops randomly within the image dimensions.
    """
    B = images.shape[0]
    # create random crop bounds within the image dimensions
    image_shape = images.shape[-2:]
    crop_shape = (crop_scale * image_shape[0], crop_scale * image_shape[1])

    if random:
        center = torch.stack([
            torch.rand(B, device=images.device, dtype=images.dtype) * (image_shape[1] - crop_shape[1]) + crop_shape[1] / 2,
            torch.rand(B, device=images.device, dtype=images.dtype) * (image_shape[0] - crop_shape[0]) + crop_shape[0] / 2,
        ], dim=1)
    else:
        center = torch.stack([
            torch.ones(B, device=images.device, dtype=images.dtype) * image_shape[1] / 2,
            torch.ones(B, device=images.device, dtype=images.dtype) * image_shape[0] / 2,
        ], dim=1)

    crop_bounds = get_crop_bounds(
        center=center,
        crop_shape=crop_shape,
    )
    crop = batched_crop(
        images=images,
        crop_bounds=crop_bounds,
        out_shape=out_shape,
    )

    # modify kp to new crop bounds
    if kp is not None:
        assert kp.ndim == 2 and kp.shape == (B, 2)
        kp = denormalize_keypoints(kp, image_size=image_shape) if kp is not None else None
        kp = torch.stack([
            kp[:, 0] - crop_bounds[:, 0, 0],
            kp[:, 1] - crop_bounds[:, 0, 1],
        ], dim=-1)
        kp = normalize_keypoints(kp, image_size=crop_shape)

    return crop, kp

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)
