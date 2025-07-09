from torch import nn, Tensor
import torch
import logging
import abc
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import einops

class VisionEncoder(abc.ABC, nn.Module):
    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Returns the embedding dimension of the encoder."""
        pass

    def num_tokens(self, height, width) -> int:
        x = torch.zeros((1, 3, height, width), dtype=torch.float32)
        return self.forward(x).shape[1]

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the encoder."""
        pass

class ResNet(VisionEncoder):
    def __init__(self, pool=True):
        super().__init__()
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)["feature_map"]
        x = self.pool(x)
        return einops.rearrange(x, "b c h w -> b (h w) c")
        
class DINO(nn.Module):
    def __init__(self, num_freeze_layers: int = 0):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

        if num_freeze_layers == 0:
            logging.info("[DINO] Not freezing any layers of backbone")
        elif num_freeze_layers < 0:
            logging.info("[DINO] Freezing all layers of backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            logging.info(f"[DINO] Freezing first {num_freeze_layers} layers of backbone")
            startswith = [f"blocks.{i}." for i in range(num_freeze_layers)]
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

    def forward(self, x: Tensor, masks: Tensor = None) -> Tensor:
        feat = self.backbone.forward_features(x, masks)
        return torch.cat([
            feat["x_norm_clstoken"].unsqueeze(1),
            feat["x_norm_patchtokens"],
        ], dim=1)
    
def get_vision_encoder(name: str, **kwargs) -> VisionEncoder:
    if name == "resnet":
        return ResNet(**kwargs)
    elif name == "dino":
        return DINO(**kwargs)
    else:
        raise ValueError(f"Unknown vision encoder: {name}")