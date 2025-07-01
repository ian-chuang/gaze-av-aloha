import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from gaze_av_aloha.utils.policy_utils import get_output_shape
from torchvision.ops import roi_align
from gaze_av_aloha.policies.gaze_policy.transformer import AttentionPooling

class VisionEncoder(nn.Module):
    def __init__(
        self, 
        resize_shape: tuple[int, int],
        crop_shape: tuple[int, int] | None = None,
        crop_is_random: bool = True,
        out_dim: int = 64,
    ):
        super().__init__()
        # pre proc
        self.resize = torchvision.transforms.Resize(resize_shape)
        if crop_shape is not None and crop_shape != resize_shape:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # backbone = torchvision.models.resnet18(
        #     weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        # )
        # for module in backbone.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.eval()  # Use running stats, not batch stats
        #         module.weight.requires_grad = False
        #         module.bias.requires_grad = False
        # self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.attn_pooling = AttentionPooling(
            hidden_size=self.backbone.embed_dim,
            out_dim=out_dim,
            num_queries=4,
            depth=2,
            num_heads=6,
            mlp_ratio=4.0,
            dropout=0.1,
            max_seq_len=512,
        ) 

        self.gaze_proj = nn.Linear(2, self.backbone.embed_dim)

        # post proc
        # if not self.do_crop:
        #     dummy_shape = (1, 3, *resize_shape)
        # else:
        #     dummy_shape = (1, 3, *crop_shape)
        # feature_map_shape = get_output_shape(lambda x: self.backbone(x)["feature_map"], dummy_shape)[1:]
        # if use_spatial_softmax:
        #     self.pool = SpatialSoftmax(feature_map_shape, num_kp=num_kp)
        #     self.out = nn.Sequential(
        #         nn.Linear(num_kp*2, out_dim),
        #         nn.GELU(),
        #     )
        # else:
        #     self.pool = nn.AdaptiveAvgPool2d((1, 1))
        #     self.out = nn.Sequential(
        #         nn.Linear(feature_map_shape[0], out_dim),
        #         nn.GELU(),
        #     )
        

    def forward(self, x: Tensor, gaze: Tensor | None = None) -> Tensor:
        x = self.resize(x)
        if self.do_crop:
            if self.training:  
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = self.backbone.forward_features(x)["x_norm_patchtokens"]

        if gaze is not None:
            x = torch.cat([
                x, 
                self.gaze_proj(gaze).unsqueeze(1)
            ], dim=1)

        x = self.attn_pooling(x)
        return x

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

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints
    
def crop_at_center(images: Tensor, centers: Tensor, crop_shape: tuple, out_shape: tuple | None = None) -> Tensor:
    # crop the images around the predicted gaze
    if out_shape is None:
        out_shape = crop_shape
    boxes = torch.zeros(centers.shape[0], 5, device=centers.device)
    h, w = images.shape[-2:]
    new_h, new_w = crop_shape
    eye_pixel_x = ((centers[:, 0] + 1) / 2) * w
    eye_pixel_y = ((centers[:, 1] + 1) / 2) * h
    boxes[:, 0] = torch.arange(centers.shape[0], device=centers.device)
    boxes[:, 1] = eye_pixel_x - new_w / 2
    boxes[:, 2] = eye_pixel_y - new_h / 2
    boxes[:, 3] = eye_pixel_x + new_w / 2
    boxes[:, 4] = eye_pixel_y + new_h / 2     
    images = roi_align(
        images,
        boxes,
        output_size=out_shape,
    )   
    return images