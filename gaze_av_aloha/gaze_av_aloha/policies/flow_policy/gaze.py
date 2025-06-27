import torch 
from torch import Tensor
import torchvision
import einops

def create_saliency_map(eye, featmap_size, sigma=0.1):
    """
    Create a saliency map for the given eye position. It is scaled according to the size of the feature map.
    Args:
        eye (torch.Tensor): The eye position in the range [-1, 1] (x, y)
        featmap_size (tuple): The size of the feature map (height, width)
        sigma_ratio (float): Ratio of sigma to the feature map size. Default is 0.1 (10% of feature map size).
    """
    h_featmap, w_featmap = featmap_size
    batch_size = eye.shape[0]
    
    # Compute sigma based on the feature map size
    sigma = sigma * (h_featmap + w_featmap) / 2.0
    
    # Generate the meshgrid for the feature map size
    x = torch.arange(w_featmap, dtype=torch.float32, device=eye.device).view(1, -1).repeat(h_featmap, 1)
    y = torch.arange(h_featmap, dtype=torch.float32, device=eye.device).view(-1, 1).repeat(1, w_featmap)
    x, y = x.unsqueeze(0), y.unsqueeze(0)  # Add batch dimension
    
    # Expand meshgrid to batch size
    x = x.repeat(batch_size, 1, 1)
    y = y.repeat(batch_size, 1, 1)

    # Convert normalized coordinates to feature map indices
    eye_x = ((eye[:, 0] + 1) / 2).view(-1, 1, 1) * w_featmap
    eye_y = ((eye[:, 1] + 1) / 2).view(-1, 1, 1) * h_featmap
    
    # Calculate the Gaussian function
    gaussian = torch.exp(-(((x - eye_x) ** 2) / (2 * sigma ** 2) +
                           ((y - eye_y) ** 2) / (2 * sigma ** 2)))

    return gaussian

def gaze_crop(images: Tensor, crop_shape: tuple, gaze: Tensor = None, crop_is_random: bool =False):
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

    if gaze is not None:
        gaze = torch.stack([
            (((gaze[..., 0]+1)/2) * image_shape[1] - j) / w * 2 - 1,
            (((gaze[..., 1]+1)/2) * image_shape[0] - i) / h * 2 - 1,
        ], dim=-1)

    return images, gaze

def gaze_mask(gaze, patch_shape, sigma=0.1, k=20, random=True):
    scores = create_saliency_map(gaze, featmap_size=patch_shape, sigma=sigma)
    scores = einops.rearrange(scores, 'b h w -> b (h w)')
    if random:
        indices = torch.multinomial(scores, num_samples=k, replacement=False)
    else:
        _, indices = torch.topk(scores, k, dim=1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    batch_idx = torch.arange(scores.shape[0], device=indices.device).unsqueeze(1).expand(-1, k)
    mask[batch_idx, indices] = False
    return mask
    
