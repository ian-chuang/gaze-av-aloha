import einops
import torch
import torchvision
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn.functional as F
from gaze_av_aloha.configs import TaskConfig
from gaze_av_aloha.policies.gaze_policy.gaze_policy_config import GazePolicyConfig
from gaze_av_aloha.policies.gaze_policy.tokenizer import (
    BaseImageTokenizer,
    FoveatedImageTokenizer,
)
from gaze_av_aloha.policies.gaze_policy.vit import create_vit_s, AttentionPooling
from torchvision.transforms import Resize, CenterCrop, RandomCrop

class BaseObserver(nn.Module):
    def __init__(
        self,
        state_key: str = 'observation.state',
        image_keys: list[str] = ['observation.image'],
        resize_shape: tuple[int, int] = (240, 320),
        crop_shape: tuple[int, int] = (224, 308),
    ):
        super().__init__()
        self.state_key = state_key
        self.image_keys = image_keys
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

    def get_states(self, batch):
        return batch[self.state_key]

    def get_images(self, batch):
        images = torch.stack([batch[key] for key in self.image_keys], dim=2)
        images = einops.rearrange(images, 'b s n c h w -> (b s n) c h w')

        if images.shape[-2:] != tuple(self.resize_shape):
            images = F.interpolate(images, size=tuple(self.resize_shape), mode='bilinear', align_corners=False)

        ch, cw = self.crop_shape
        _, _, H, W = images.shape
        top = torch.randint(0, H - ch + 1, (1,)).item()
        left = torch.randint(0, W - cw + 1, (1,)).item()
        images = images[..., top:top+ch, left:left+cw]
        return images

    def observe(self, batch):
        raise NotImplementedError("observe() should be implemented.")

    def forward(self, batch):
        features = self.observe(batch)
        return features

class ResNetObserver(BaseObserver):
    """
    Extracts 1D or 2D embeddings from a ResNet backbone and concat with states.

    When tokenize=False: Returns concatenated 1D features
    When tokenize=True: Returns 2D token features where ResNet features are spatial tokens
                       and state is added as an additional token.
    """

    def __init__(
        self,
        state_key: str = 'observation.state',
        image_keys: list[str] = ['observation.image'],
        resize_shape: tuple[int, int] = (240, 320),
        crop_shape: tuple[int, int] = (224, 308),
        state_dim: int = 21,
        tokenize: bool = False,
    ):
        super().__init__(
            state_key=state_key,
            image_keys=image_keys,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
        )

        self.tokenize = tokenize
        self.state_dim = state_dim

        # ResNet backbone setup
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})

        if self.tokenize:
            self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
            self.state_projector = nn.Linear(state_dim, 512)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def observe(self, batch):
        if self.tokenize:
            return self._observe_tokenized(batch)
        else:
            return self._observe_flattened(batch)

    def _observe_tokenized(self, batch):
        b = batch[self.state_key].shape[0]
        s = batch[self.image_keys[0]].shape[1]
        n = len(self.image_keys)

        images = self.get_images(batch)
        img_features = self.backbone(images)["feature_map"]
        img_features = self.spatial_pool(img_features)

        img_tokens = einops.rearrange(img_features, 'bsn c h w -> bsn (h w) c')

        img_tokens = einops.rearrange(
            img_tokens, '(b s n) hw c -> b (s n hw) c', b=b, s=s, n=n
        )

        states = self.get_states(batch).flatten(start_dim=1)
        state_tokens = self.state_projector(states).unsqueeze(1)

        tokens = torch.cat([state_tokens, img_tokens], dim=1)

        return tokens

    def _observe_flattened(self, batch):
        features = []
        features.append(self.get_states(batch).flatten(start_dim=1))

        b = batch[self.state_key].shape[0]
        s = batch[self.state_key].shape[1]
        n = len(self.image_keys)

        images = self.get_images(batch)
        img_features = self.pool(self.backbone(images)["feature_map"])
        features.append(
            einops.rearrange(
                img_features, '(b s n) c h w -> b (s n c h w)', b=b, s=s, n=n
            )
        )

        features = torch.cat(features, dim=1)
        return features
    
class ViTObserver(nn.Module):
    def __init__(
        self,
        policy_cfg: GazePolicyConfig,
        task_cfg: TaskConfig,
    ):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg

        n_tokens = 0
        n_obs_steps = policy_cfg.n_obs_steps
        n_images = len(policy_cfg.image_to_gaze_key)

        self.state_proj = nn.Sequential(
            nn.Dropout(policy_cfg.dropout),
            nn.Linear(task_cfg.state_dim, policy_cfg.dim_model),
        )
        n_tokens += n_obs_steps

        if policy_cfg.use_gaze:
            self.tokenizer = FoveatedImageTokenizer(
                token_size=policy_cfg.patch_size,
                strides=policy_cfg.strides,
                grid_sizes=policy_cfg.grid_sizes,
                height=policy_cfg.vit_input_shape[0],
                width=policy_cfg.vit_input_shape[1],
            )
        else:
            self.tokenizer = BaseImageTokenizer(
                token_size=policy_cfg.patch_size,
                height=policy_cfg.vit_input_shape[0],
                width=policy_cfg.vit_input_shape[1],
            )
        self.vit = create_vit_s(self.tokenizer.get_num_tokens(), self.tokenizer.get_token_size())
        n_tokens += self.tokenizer.get_num_tokens() * n_obs_steps * n_images

        # self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        # h, w = policy_cfg.vit_input_shape[0] // self.vit.patch_size, policy_cfg.vit_input_shape[0] // self.vit.patch_size
        # n_tokens += (h * w) * n_obs_steps * n_images

        self.vit_proj = nn.Linear(self.vit.embed_dim, policy_cfg.dim_model)
        
        

        if policy_cfg.use_gaze:
            self.gaze_proj = nn.Sequential(
                nn.Dropout(policy_cfg.dropout),
                nn.Linear(2, policy_cfg.dim_model),
            )
            n_tokens += policy_cfg.n_obs_steps * n_images
        
        self.pool = AttentionPooling(
            hidden_size=policy_cfg.dim_model,
            num_queries=policy_cfg.pool_n_queries,
            depth=policy_cfg.pool_n_layers,
            num_heads=policy_cfg.num_heads,
            mlp_ratio=policy_cfg.mlp_ratio,
            dropout=policy_cfg.dropout,
        ) 
        self.pool_pos_embed = nn.Embedding(n_tokens, policy_cfg.dim_model)

    def forward(self, batch):
        viz = {}
        tokens = []
        pos_embed = []
        batch = dict(batch)  

        # load image and resize to input shape
        img = torch.stack([batch[key] for key in self.cfg.image_to_gaze_key.keys()], dim=2)
        b, s, n = img.shape[:3]
        img = einops.rearrange(img, 'b s n c h w -> (b s n) c h w')  
        img = Resize(self.cfg.input_shape)(img)  

        # state
        tokens.append(self.state_proj(batch[self.task_cfg.state_key]))

        # foveal vision
        if self.cfg.use_gaze:
            gaze = torch.stack([batch[key] for key in self.cfg.image_to_gaze_key.values()], dim=2)
            gaze = einops.rearrange(gaze, 'b s n c -> (b s n) c') 
            if self.training:
                gaze = gaze + torch.randn_like(gaze) * self.cfg.gaze_noise
            tokens.append(
                einops.rearrange(
                    self.gaze_proj(gaze),  # send in the gaze (frame of ref of robot camera)
                    '(b s n) d -> b (s n) d',
                    b=b, s=s, n=n
                ) 
            )
        else:
            if self.cfg.use_crop:
                if self.training:
                    img = RandomCrop(self.cfg.crop_shape)(img)  # random crop if not using gaze
                else:
                    img = CenterCrop(self.cfg.crop_shape)(img)
            gaze = None
        
        patch_tokens, masks = self.tokenizer.tokenize(img, gaze)
        if not self.training:
            viz["patch_tokens"] = self.tokenizer.generate_visualization(patch_tokens[0]).unsqueeze(0)  # add batch dimension for visualization
        features, reg_tokens = self.vit(patch_tokens, masks)

        # img = Resize(self.cfg.vit_input_shape)(img)
        # features = self.vit.forward_features(img)["x_norm_patchtokens"]

        tokens.append(
            einops.rearrange(
                self.vit_proj(features),  
                '(b s n) l d -> b (s n l) d',   
                b=b, s=s, n=n
            )
        )

        tokens = torch.cat(tokens, dim=1)
        pos_embed = self.pool_pos_embed.weight.unsqueeze(0)
        cond = self.pool(tokens, pos_embed)
        return cond
