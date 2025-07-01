from torch import nn, Tensor
import torch
import logging
from pprint import pformat

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
                    logging.info(f"[DINO] Freezing parameter: {name}")

    @property
    def embed_dim(self):
        return self.backbone.embed_dim
    
    @property
    def patch_size(self):
        return self.backbone.patch_size

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone.forward_features(x)['x_norm_patchtokens']