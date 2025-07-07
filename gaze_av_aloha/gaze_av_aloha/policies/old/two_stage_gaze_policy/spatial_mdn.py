import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from collections import OrderedDict
from torchvision.models.resnet import conv1x1, conv3x3
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
from typing import Optional, Callable, Type
from torchvision.ops.misc import FrozenBatchNorm2d
from typing import List, Optional, Sequence
from torchvision.ops import roi_align
import re
import numpy as np
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from torch.distributions.distribution import Distribution
import einops
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torchvision.transforms import Resize
from gaze_av_aloha.policies.gaze_policy.resnet import FiLMResNet18Unet

# project to closest positive definite matrix
def cholesky(matrix: Tensor) -> Tensor:
    eigval, eigvec = torch.linalg.eigh(matrix)
    eigval = torch.clamp(eigval, min=1e-6)  # Ensure eigenvalues are positive
    matrix = eigvec @ torch.diag_embed(eigval) @ eigvec.transpose(-1, -2)
    return torch.linalg.cholesky(matrix)

class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        B, K = features.shape[:2]

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        mean = attention @ self.pos_grid  # (B*K, 2)

        pos_outer = self.pos_grid[:, :, None] @ self.pos_grid[:, None, :]
        E_xxT = torch.einsum('bn,nij->bij', attention, pos_outer)
        mean_outer = mean[:, :, None] @ mean[:, None, :]
        cov = E_xxT - mean_outer

        # Reshape
        mean = mean.view(B, K, 2)
        cov = cov.view(B, K, 2, 2)

        return mean, cov

class DistributionHeatmap(nn.Module):
    def __init__(self, heatmap_shape=(48, 64)):
        super().__init__()
        H, W = heatmap_shape
        # Create meshgrid of shape (H, W)
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)
        grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        grid = einops.rearrange(grid, 'h w d -> (h w) 1 d')
        self.register_buffer('grid', grid)  
        
        self.H = H
        self.W = W

    def forward(self, dist: Distribution) -> Tensor:
        assert len(dist.batch_shape) == 1, "Distribution must have a batch shape of 1"
        B = dist.batch_shape[0]
        grid = self.grid.expand(-1, B, -1)
        probs = dist.log_prob(grid).exp()
        probs = einops.rearrange(probs, '(h w) b -> b h w', h=self.H, w=self.W)
        return probs

def gmm_highest_probability_mean(gmm: MixtureSameFamily) -> Tensor:
    # Get the most probable component index for each batch
    topk_indices = gmm.mixture_distribution.probs.argmax(dim=-1)  # shape (B,)
    means = gmm.component_distribution.mean  # shape (B, K, D)
    B = means.shape[0]
    mean = means[torch.arange(B), topk_indices]  # shape (B, D)
    return mean

def gmm_average_entropy(gmm: MixtureSameFamily) -> Tensor:
    """
    Return the average entropy across all Gaussian components for each batch item.
    """
    component_entropies = gmm.component_distribution.entropy()  # (B, K)
    mix_probs = gmm.mixture_distribution.probs  # (B, K)
    # Weighted average entropy across components
    avg_entropy = (mix_probs * component_entropies).sum(dim=-1).mean()
    return avg_entropy

class SpatialMDN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        resize_shape=(48, 64),
        unet_cond_dim = 768,
        unet_pretrained=True,
        unet_n_freeze_layers=0,
        num_components=8,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.unet_cond_dim = unet_cond_dim
        
        self.unet = FiLMResNet18Unet(
            num_classes=num_components,
            cond_dim=unet_cond_dim,
            n_freeze_layers=unet_n_freeze_layers,
            pretrained=unet_pretrained,
        )
        self.spatial_softmax = SpatialSoftmax(input_shape=(num_components, resize_shape[0], resize_shape[1]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling to get a feature vector
        self.weight_out = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_components),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Distribution:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
            cond: Conditioning tensor of shape (B, unet_cond_dim).
        """
        assert cond.shape[1] == self.unet_cond_dim, f"Conditioning dimension mismatch: expected {self.unet_cond_dim}, got {cond.shape[1]}"
        
        x = Resize(self.resize_shape)(x)  # Resize input to (B, C, H, W)
        heatmap, feat = self.unet(x, cond=cond, ret_feat=True) # (B, num_components, H, W), (B, 512, H, W)
        mean, cov = self.spatial_softmax(heatmap) # (B, N, 2), (B, N, 2, 2)
        weight = self.weight_out(self.pool.forward(feat).flatten(start_dim=1))  # (B, num_components)
        # Construct mixture model
        mixture = Categorical(logits=weight)
        components = MultivariateNormal(loc=mean, scale_tril=cholesky(cov))  # (B, N, 2, 2)
        return MixtureSameFamily(mixture, components)
