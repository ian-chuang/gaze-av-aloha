from torch import nn, Tensor
import torch
import logging
import abc
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import einops
from typing import List, Optional, Tuple
from torchvision.transforms import CenterCrop, RandomCrop
from gaze_av_aloha.policies.gaze_policy.vit import create_vit_s, create_vit_b
from gaze_av_aloha.policies.gaze_policy.tokenizer import FoveatedImageTokenizer, BaseImageTokenizer

class VisionEncoder(abc.ABC, nn.Module):
    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Returns the embedding dimension of the encoder."""
        pass
    
    def get_num_tokens(self, height: int, width: int, device="cpu") -> int:
        """Returns the number of tokens produced by the encoder."""
        x = torch.randn(1, 3, height, width, device=device)
        centers = torch.zeros((1, 2), device=x.device)  # Dummy centers
        tokens, _ = self.forward(x, centers)
        return tokens.shape[1]

    @abc.abstractmethod
    def forward(self, x: Tensor, centers: Tensor=None, visualize: bool = False) -> Tuple[Tensor, dict]:
        """Forward pass of the encoder."""
        pass

class ResNet(VisionEncoder):
    def __init__(
        self, 
        crop_shape: Tuple[int, int] = (216, 288),
        pool: bool = True,
    ):
        super().__init__()
        self.random_crop = RandomCrop(crop_shape)
        self.center_crop = CenterCrop(crop_shape)
        backbone_model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        self._embed_dim = backbone_model.fc.in_features
        if pool:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.Identity()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: Tensor, centers: Tensor=None, visualize: bool = False) -> Tuple[Tensor, dict]:
        viz = {}

        if self.training:
            x = self.random_crop(x)
        else:
            x = self.center_crop(x)

        if not self.training and visualize:
            viz["input"] = x

        x = self.backbone(x)["feature_map"]
        x = self.pool(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        return x, viz
        
class DINO(VisionEncoder):
    def __init__(
        self, 
        n_freeze_layers: int = 0,
        crop_shape: Tuple[int, int] = (216, 288),
    ):
        super().__init__()
        self.random_crop = RandomCrop(crop_shape)
        self.center_crop = CenterCrop(crop_shape)

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

        if n_freeze_layers == 0:
            logging.info("[DINO] Not freezing any layers of backbone")
        elif n_freeze_layers < 0:
            logging.info("[DINO] Freezing all layers of backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            logging.info(f"[DINO] Freezing first {n_freeze_layers} layers of backbone")
            startswith = [f"blocks.{i}." for i in range(n_freeze_layers)]
            startswith += [
                "cls_token",
                "pos_embed",
                "register_tokens",
                "mask_token",
                "patch_embed.proj.weight",
                "patch_embed.proj.bias",
            ]
            for name, param in self.backbone.named_parameters():
                if any([name.startswith(s) for s in startswith]):
                    param.requires_grad = False

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    def forward(self, x: Tensor, centers: Tensor=None, visualize: bool = False) -> Tuple[Tensor, dict]:
        viz = {}

        if self.training:
            x = self.random_crop(x)
        else:
            x = self.center_crop(x)
        
        if not self.training and visualize:
            viz["input"] = x

        x = self.backbone.forward_features(x)["x_norm_patchtokens"]

        return x, viz
    
class ViT(VisionEncoder):
    def __init__(
        self, 
        repo_id: Optional[str] = "",
        crop_shape: Tuple[int, int] = (216, 288),
        token_size: int = 16,
        height: int = 224,
        width: int = 224,
    ):
        super().__init__()
        self.random_crop = RandomCrop(crop_shape)
        self.center_crop = CenterCrop(crop_shape)

        self.tokenizer = BaseImageTokenizer(
            token_size=token_size,
            height=height,
            width=width,
        )
        self.backbone = create_vit_b(
            self.tokenizer.get_num_tokens(),
            self.tokenizer.get_token_size()
        )
        if repo_id:
            logging.info(f"Loading vit weights from {repo_id}")
            self.backbone = self.backbone.from_pretrained(repo_id)

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    def forward(self, x: Tensor, centers: Tensor=None, visualize: bool = False) -> Tuple[Tensor, dict]:
        viz = {}

        if self.training:
            x = self.random_crop(x)
        else:
            x = self.center_crop(x)

        patch_tokens, _ = self.tokenizer.tokenize(x)
        if not self.training and visualize:
            viz["input"] = self.tokenizer.generate_visualization(
                patch_tokens[0]
            ).unsqueeze(0)

        features, reg_tokens = self.backbone(patch_tokens)

        assert features.shape[1] == self.tokenizer.get_num_tokens(), \
            f"Expected {self.tokenizer.get_num_tokens()} tokens, got {features.shape[1]}"

        return features, viz

class FoveatedViT(VisionEncoder):
    def __init__(
        self, 
        repo_id: str = "",
        token_size: int = 16, 
        strides: List[int] = [1, 2, 6], 
        grid_sizes: List[int] = [2,3,3], 
        height: int = 288, 
        width: int = 288
    ):
        super().__init__()
        self.tokenizer = FoveatedImageTokenizer(
            token_size=token_size, 
            strides=strides, 
            grid_sizes=grid_sizes, 
            height=height, 
            width=width
        )
        self.backbone = create_vit_b(
            self.tokenizer.get_num_tokens(), self.tokenizer.get_token_size()
        )
        if repo_id:
            logging.info(f"Loading vit weights from {repo_id}")
            self.backbone = self.backbone.from_pretrained(repo_id)

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim
    
    def forward(self, x: Tensor, centers: Tensor=None, visualize: bool = False) -> Tuple[Tensor, dict]:
        viz = {}

        patch_tokens, masks = self.tokenizer.tokenize(x, centers)

        if not self.training and visualize:
            viz["input"] = self.tokenizer.generate_visualization(
                patch_tokens[0]
            ).unsqueeze(0)  # add batch dimension for visualization

        features, reg_tokens = self.backbone(patch_tokens, masks)

        assert features.shape[1] == self.tokenizer.get_num_tokens(), \
            f"Expected {self.tokenizer.get_num_tokens()} tokens, got {features.shape[1]}"

        return features, viz

def get_vision_encoder(name: str, **kwargs) -> VisionEncoder:
    if name == "resnet":
        return ResNet(**kwargs)
    elif name == "dino":
        return DINO(**kwargs)
    elif name == "vit":
        return ViT(**kwargs)
    elif name == "foveated_vit":
        return FoveatedViT(**kwargs)
    else:
        raise ValueError(f"Unknown vision encoder: {name}")