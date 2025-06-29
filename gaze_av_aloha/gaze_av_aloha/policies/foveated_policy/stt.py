from torch import nn, Tensor
import torch
from torchvision.ops import roi_align

from segment_this_thing import build_segment_this_thing_b
from segment_this_thing.foveation import generate_grid_coords_2d, Foveator
from segment_this_thing.model import extend_valid_token_mask

import gaze_av_aloha
from gaze_av_aloha.policies.foveated_policy.utils import denormalize_keypoints

from pathlib import Path
import os
from tqdm import tqdm
import logging


STT_URL = "https://huggingface.co/facebook/segment_this_thing/resolve/main/stt-b-qbkbmb5qsb4q2.pth"
STT_PATH = Path(os.path.dirname(os.path.dirname(gaze_av_aloha.__file__))) / "cache" / "stt-b.pth"

def download_file(url: str, filename: str):
    """
    Downloads a file from a given URL to a specified filename.
    If the file does not exist, it will create the necessary directories.
    Args:
        url (str): The URL to download the file from.
        filename (str): The path where the file should be saved.
    """
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte
            with open(filename, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")

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

def get_crop_bounds(center: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Calculates the lower and upper bounds for cropping an image centered around a given point.
    Args:
        center (torch.Tensor): The center point for the crop, shape [B, 2].
        crop_size (int): The size of the crop to be applied.
    Returns:
        torch.Tensor: A tensor containing the lower and upper corners of the crop, shape [B, 2, 2].
    """
    lower_corner = (center - crop_size / 2).floor()
    upper_corner = lower_corner + crop_size
    return torch.stack([lower_corner, upper_corner], dim=1)  

def compute_integral_image(image: torch.Tensor):
    """
    Computes the integral image of the given input image.
    The integral image is computed by cumulatively summing the pixel values along both dimensions.
    Args:
        image (torch.Tensor): Input image tensor of shape [B, C, H, W].
    Returns:
        torch.Tensor: The integral image tensor of shape [B, C, H, W].
    """
    padded = torch.nn.functional.pad(image, (1, 0, 1, 0), mode="constant", value=0)
    return padded.cumsum(dim=3).cumsum(dim=2)

class BatchedFoveator(Foveator):
    """
    A batched version of the Foveator class that allows for processing multiple images in a batch.
    This class extends the Foveator class and overrides the extract_foveated_image method to handle batches.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "grid_coords_2d",
            generate_grid_coords_2d(
                self.token_size
            ).unsqueeze(0)
        )

    def extract_foveated_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        # This function extracts a set of tokens representing a foveated version of the input image.
        # Input: images (Tensor of shape [B, C, H, W]).
        # Output: foveated tokens (Tensor of shape [B, N, C, H, W])
        """
        if not images.ndim == 4:
            raise ValueError(
                "[Foveator.extract_foveated_image]: Expected 3D input Tensor."
            )

        expected_input_size = self.token_size * self.block_sizes[-1]
        if (
            images.shape[-2] != expected_input_size
            or images.shape[-1] != expected_input_size
        ):
            raise ValueError(
                f"[Foveator.extract_foveated_image]: Expected square image of size {expected_input_size}"
            )
        if images.shape[-3] != 3:
            raise ValueError(
                f"[Foveator.extract_foveated_image]: Expected 3-channel image."
            )
        
        integral_image = compute_integral_image(images) # (B, C, H, W)

        # self.token_corner_indices is (N, U)
        # self.token_strides is (N)
        # generate_grid_coords_2d return (H, W, U)
        # All get mapped to (N, H, W, U)
        lower_pixel_coords = self.token_corner_indices.view(
            -1, 1, 1, 2
        ) + self.token_strides.view(-1, 1, 1, 1) * self.grid_coords_2d
        upper_pixel_coords = lower_pixel_coords + self.token_strides.view(-1, 1, 1, 1)

        # lol
        B, C, H, W = integral_image.shape
        N, I1, I2, _ = lower_pixel_coords.shape
        lower_pixel_coords = lower_pixel_coords.unsqueeze(1).unsqueeze(0).expand(B, N, C, I1, I2, 2)
        upper_pixel_coords = upper_pixel_coords.unsqueeze(1).unsqueeze(0).expand(B, N, C, I1, I2, 2)
        b_idx = torch.arange(B).view(B, 1, 1, 1, 1).expand(B, N, C, I1, I2)
        c_idx = torch.arange(C).view(1, 1, C, 1, 1).expand(B, N, C, I1, I2)

        return (
            integral_image[
                b_idx, c_idx,
                upper_pixel_coords[..., 1], upper_pixel_coords[..., 0]
            ]
            - integral_image[
                b_idx, c_idx,
                upper_pixel_coords[..., 1], lower_pixel_coords[..., 0]
            ]
            - integral_image[
                b_idx, c_idx,
                lower_pixel_coords[..., 1], upper_pixel_coords[..., 0]
            ]
            + integral_image[
                b_idx, c_idx,
                lower_pixel_coords[..., 1], lower_pixel_coords[..., 0]
            ]
        ) / self.token_strides.square().view(1, -1, 1, 1, 1).float()

    def get_in_bounds_tokens(
        self,
        image_size: torch.Tensor,           
        crop_bounds: torch.Tensor,         
        in_bounds_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Returns a binary mask indicating which tokens are sufficiently in bounds to be considered valid.
        
        Args:
            image_size: Tensor of shape [2], the size of the input image (width, height).
            crop_bounds: Tensor of shape [B, 2, 2], where each entry is [[x1, y1], [x2, y2]] for each batch.
            in_bounds_threshold: Minimum fraction of token area that must be in bounds.
            
        Returns:
            valid_token_mask: Tensor of shape [B, N] indicating valid tokens per batch element.
        """
        device = crop_bounds.device
        image_size = image_size.to(device)

        # (B, N, 2) = (1, N, 2) + (B, 1, 2)
        foveation_offset = crop_bounds[:, 0]  # (B, 2)
        token_corner_coordinates = self.token_corner_indices[None, :, :] + foveation_offset[:, None, :]  # (B, N, 2)

        bounded_lower_corners = token_corner_coordinates.clamp(min=0)
        bounded_upper_corners = torch.minimum(
            token_corner_coordinates + self.token_size * self.token_strides.view(1, -1, 1),  # (1, N, 2)
            image_size.view(1, 1, 2),
        )

        area_per_pixel = self.token_strides.square()  # (N,)
        in_bounds_area = (
            (bounded_upper_corners - bounded_lower_corners).clamp(min=0).prod(dim=-1).float()
        )  # (B, N)

        valid_token_mask = (in_bounds_area / area_per_pixel.float()) > in_bounds_threshold  # (B, N)
        return valid_token_mask

class STTEncoder(nn.Module):
    def __init__(self, freeze_n_layers: int = 0):
        super().__init__()
        download_file(STT_URL, STT_PATH)
        self.foveator = BatchedFoveator(token_size=16, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10])
        stt = build_segment_this_thing_b(
            num_tokens=self.foveator.get_num_tokens(),
            token_size=16
        )
        stt.load_state_dict(torch.load(STT_PATH, weights_only=True))
        self.patch_emb = stt.image_encoder.patch_emb
        self.pos_enc = stt.image_encoder.pos_enc
        self.reg_tokens = stt.image_encoder.reg_tokens
        self.blocks = stt.image_encoder.blocks

        if freeze_n_layers == 0:
            logging.info("[STTEncoder] Not freezing any layers of backbone")
        elif freeze_n_layers < 0 or freeze_n_layers >= len(self.blocks):
            logging.info("[STTEncoder] Freezing all layers of backbone")
            for param in self.parameters():
                param.requires_grad = False
        else:
            logging.info(f"[STTEncoder] Freezing first {freeze_n_layers} layers of backbone")
            for param in self.patch_emb.parameters():
                param.requires_grad = False
            self.pos_enc.requires_grad = False
            self.reg_tokens.requires_grad = False
            for block in self.blocks[:freeze_n_layers]:
                for param in block.parameters():
                    param.requires_grad = False

    @property
    def num_patch_tokens(self) -> int:
        return self.foveator.get_num_tokens()
    
    @property
    def num_register_tokens(self) -> int:
        return self.reg_tokens.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.reg_tokens.shape[1]
    
    @property
    def input_size(self) -> int:
        return self.foveator.get_pattern_bounds_size()

    def prepare_patches_and_mask(self, images: Tensor, centers: Tensor) -> Tensor:
        """
        prepare_tokens extracts foveated tokens from the input images based on the provided centers.
        Args:
            images (Tensor): Input images of shape [B, C, H, W].
            centers (Tensor): Foveation centers of shape [B, 2], Normalized to [-1, 1].
        Returns:
            Tensor: Foveated tokens of shape [B, N, C, H, W].
        """
        size = self.input_size
        h, w = images.shape[-2:]
        centers = denormalize_keypoints(centers, (h, w))
        crop_bounds = get_crop_bounds(centers, size)
        crops = batched_crop(images, crop_bounds, out_shape=(size, size))
        mask = self.foveator.get_in_bounds_tokens(
            torch.tensor([w, h], device=images.device),
            crop_bounds
        )
        patches = self.foveator.extract_foveated_image(crops)
        return patches, mask

    def forward(self, images: Tensor, centers: Tensor) -> Tensor:
        patches, mask = self.prepare_patches_and_mask(images, centers)

        embedded_tokens = self.patch_emb(
            patches.flatten(2)
        ) + self.pos_enc.unsqueeze(0)

        num_registers = self.reg_tokens.shape[0]

        embedded_tokens = torch.cat(
            [
                embedded_tokens,
                self.reg_tokens.unsqueeze(0).expand(embedded_tokens.shape[0], -1, -1),
            ],
            dim=1,
        )

        valid_token_mask = extend_valid_token_mask(mask, num_registers)

        invalid_token_mask = (
            valid_token_mask.logical_not() if valid_token_mask is not None else None
        )

        transformed_tokens = embedded_tokens
        for block in self.blocks:
            transformed_tokens = block(transformed_tokens, invalid_token_mask)

        return transformed_tokens

# Only for testing purposes lol
class STT(STTEncoder):
    def __init__(self):
        super().__init__()
        stt = build_segment_this_thing_b(
            num_tokens=self.foveator.get_num_tokens(),
            token_size=16
        )
        stt.load_state_dict(torch.load(STT_PATH, weights_only=True))
        self.image_encoder = stt.image_encoder
        self.mask_decoder = stt.mask_decoder

    def forward(self, images: Tensor, centers: Tensor) -> Tensor:
        patches, mask = self.prepare_patches_and_mask(images, centers)
        image_features, register_features = self.image_encoder(
            patches, mask
        )
        masks, ious = self.mask_decoder(image_features, register_features, mask)
        return masks, ious, patches

if __name__ == "__main__":
    import requests
    from io import BytesIO
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    batch_size = 4
    image_shape = (1280, 1280)

    stt = STT().cuda()
    encoder = STTEncoder().cuda()

    image_url = "https://picsum.photos/id/20/367/267"

    # Download image
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download image from {image_url}")
    
    # Convert image to tensor
    images = torch.tensor(
        np.array(image).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()

    # Resize images to the desired shape
    images = nn.functional.interpolate(
        images, size=image_shape, mode='bilinear', align_corners=False
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    images = (images - mean) / std  # Normalize the images

    # Generate random centers
    centers = torch.rand(batch_size, 2).cuda() * 2 - 1

    masks, ious, foveations = stt(images, centers)
    foveations = (foveations * std + mean)

    # create plt figure
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5 * batch_size))
    fig.tight_layout()
    for i in range(batch_size):
        mask = masks[i]
        iou = ious[i]
        foveation = foveations[i] * 255

        k = iou.argmax().item()
        iou = ious[k : k + 1]
        mask = mask[k : k + 1]

        num_masks = len(mask)
        num_plots = 1 + num_masks

        for m in mask:
            segmentation = stt.foveator.generate_foveated_visualization(
                m.unsqueeze(1)
            ).sigmoid()
            recon = stt.foveator.generate_foveated_visualization(foveation)
            axes[i].imshow(
                torch.where(
                    segmentation > 0.5,
                    (
                        0.5 * recon.float()
                        + 0.5 * torch.tensor([0x32, 0xA8, 0x52]).view(3, 1, 1)
                    ),
                    recon,
                )
                .permute(1, 2, 0)
                .byte()
                .cpu()
                .numpy(),
            )
    # save fig 
    fig.savefig("stt_visualization.png", bbox_inches='tight', dpi=300)

