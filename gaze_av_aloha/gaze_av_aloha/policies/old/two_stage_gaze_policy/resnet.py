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

class FiLMBlock(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_channels*2)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.film[-1].weight, 0)
        nn.init.constant_(self.film[-1].bias, 0)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        film_out = self.film(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C) -> (B, C, 1, 1)
        scale, shift = film_out.chunk(2, dim=1)
        return x * (1 + scale) + shift
    
class FiLMBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        cond_dim: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.film1 = FiLMBlock(cond_dim, planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out, cond)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class FiLMResNet18(nn.Module):
    def __init__(
            self, 
            cond_dim: int, 
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            pretrained: bool = True,
        ) -> None:
        super().__init__()
        self.inplanes = 64
        self.cond_dim = cond_dim
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = FiLMBasicBlock
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        copy_bn = self._norm_layer in [FrozenBatchNorm2d, nn.BatchNorm2d]

        # Copy weights where names match
        film_state = self.state_dict()
        pretrained_state = pretrained.state_dict()
        for name in film_state:
            # check bn[number] and film not in name
            if not copy_bn and re.search(r"bn\d+", name):
                continue
            if re.search(r"film", name):
                continue
            film_state[name] = pretrained_state[name]
        self.load_state_dict(film_state)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                cond_dim=self.cond_dim,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    cond_dim=self.cond_dim,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward_features(self, x: Tensor, cond: Tensor) -> Tensor:
        features = [x]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, cond)
            features.append(x)
        return features
    
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.forward_features(x, cond)
        return x[-1]

    @property
    def embed_dim(self) -> int:
        """Output dimension of the model."""
        return 512
    
class FiLMUnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels + skip_channels, out_channels)
        self.norm1 = norm_layer(out_channels)
        self.film1 = FiLMBlock(cond_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = norm_layer(out_channels)

        self.residual_conv = (
            conv1x1(in_channels + skip_channels, out_channels) if in_channels + skip_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        cond: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)

        identity = feature_map
        out = self.conv1(feature_map)
        out = self.norm1(out)
        out = self.film1(out, cond)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += self.residual_conv(identity)
        out = self.relu(out)

        return out

class FiLMUnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = [3, 64, 64, 128, 256, 512],
        decoder_channels: Sequence[int] = [256, 128, 64, 32, 16],
        num_classes: int = 1,
        cond_dim: int = 128,
        n_blocks: int = 5,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = FiLMUnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                cond_dim,
                norm_layer=norm_layer,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

        self.conv2d = conv3x3(
            out_channels[-1], num_classes
        )

    def forward(self, features: List[torch.Tensor], cond) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skip_connections = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, cond, height, width, skip_connection=skip_connection)

        return self.conv2d(x)
    
class FiLMResNet18Unet(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        n_freeze_layers: int = 0,
        num_classes: int = 1,
        cond_dim: int = 128,
    ):
        super().__init__()
        assert n_freeze_layers >= 0 or n_freeze_layers <= 5, "n_freeze_layers must be between 0 and 3"
        self.encoder = FiLMResNet18(
            cond_dim=cond_dim,
            norm_layer=FrozenBatchNorm2d,
            pretrained=pretrained,
        )
        if n_freeze_layers > 0:
            layers = [
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.relu,
                self.encoder.maxpool,
                self.encoder.layer1,
                self.encoder.layer2,
                self.encoder.layer3,
                self.encoder.layer4,
            ]
            layers = layers[:3 + n_freeze_layers]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.decoder = FiLMUnetDecoder(
            encoder_channels=[3, 64, 64, 128, 256, 512],
            decoder_channels=[256, 128, 64, 32, 16],
            num_classes=num_classes,
            cond_dim=cond_dim,
            n_blocks=5,
            norm_layer=lambda x: nn.GroupNorm(num_groups=x//16, num_channels=x),
            interpolation_mode="nearest",
        )

    def forward(self, x: Tensor, cond: Tensor, ret_feat=False) -> Tensor:
        features = self.encoder.forward_features(x, cond)
        logits = self.decoder(features, cond)
        if ret_feat:
            return logits, features[-1]
        return logits
