from torch import nn, Tensor
from gaze_av_aloha.policies.flow_policy.flow_policy_config import FlowPolicyConfig
from gaze_av_aloha.configs import TaskConfig
from gaze_av_aloha.policies.policy import Policy
import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from gaze_av_aloha.policies.normalize import Normalize, Unnormalize
from collections import deque
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
    get_output_shape,
)
import torchcfm.conditional_flow_matching as cfm
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
import math
import einops
# from gaze_av_aloha.models.adaln_attention import AdaLNHybridAttentionBlock, AdaLNFinalLayer
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision
import numpy as np
from typing import Callable
from torchvision.ops import roi_align
import logging

class FlowPolicy(Policy):
    def __init__(
        self, 
        policy_cfg: FlowPolicyConfig,
        task_cfg: TaskConfig,
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg
        self.stats = stats

        self.normalize_inputs = Normalize(
            {
                **{k: policy_cfg.image_norm_mode for k in task_cfg.image_keys},
                task_cfg.state_key: policy_cfg.state_norm_mode,
            },
            stats,
        )
        self.normalize_targets = Normalize(
            {
                task_cfg.action_key: policy_cfg.action_norm_mode,
            },
            stats,
        )
        self.unnormalize_outputs = Unnormalize(
            {
                task_cfg.action_key: policy_cfg.action_norm_mode,
            },
            stats,
        )
        self.flow = FlowModel(policy_cfg, task_cfg)

        self.temporal_ensembler = TemporalEnsembler(
            temporal_ensemble_coeff=policy_cfg.action_temporal_ensemble_coeff,
            chunk_size=policy_cfg.horizon,
        )

        self.reset()

    def get_optimizer(self) -> torch.optim.Optimizer:
        logging.info(f"""
            [FlowWMPolicy] Initializing AdamW optimizer with the following parameters:
            - Learning Rate: {self.cfg.optimizer_lr}
            - Learning Rate for Backbone: {self.cfg.optimizer_lr_backbone}
            - Betas: {self.cfg.optimizer_betas}
            - Epsilon: {self.cfg.optimizer_eps}
            - Weight Decay: {self.cfg.optimizer_weight_decay}
        """)
        backbone_condition = lambda n: "dino" in n
        if not any(backbone_condition(n) for n, p in self.named_parameters()):
            raise ValueError(f"No parameters found satifying the condition for vision backbone: {backbone_condition}")
        for n, p in self.named_parameters():
            if backbone_condition(n) and p.requires_grad:
                logging.info(f"{n}")
        params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not backbone_condition(n) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if backbone_condition(n) and p.requires_grad
                ],
                "lr": self.cfg.optimizer_lr_backbone,
            },
        ]
        return torch.optim.AdamW(
            params=params,
            lr=self.cfg.optimizer_lr,
            betas=self.cfg.optimizer_betas,
            eps=self.cfg.optimizer_eps,
            weight_decay=self.cfg.optimizer_weight_decay
        )
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR | None:
        return get_scheduler(
            name=self.cfg.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def get_ema(self):
        return EMAModel(
            parameters=self.parameters(),
            ema_decay=self.cfg.ema_decay,
        ) if self.cfg.use_ema else None
    
    def get_delta_timestamps(self):
        observation_indices = list(reversed(range(0, -self.cfg.n_obs_steps*self.cfg.obs_step_size, -self.cfg.obs_step_size)))
        assert len(observation_indices) == self.cfg.n_obs_steps, "Observation indices length mismatch"
        action_indices = list(range(self.cfg.horizon))
        logging.info(f"Observation indices: {observation_indices}, Action indices: {action_indices}")
        return {
            # observations
            **{
                k: [i / self.task_cfg.fps for i in observation_indices]
                for k in self.task_cfg.image_keys
            },
            self.task_cfg.state_key: [i / self.task_cfg.fps for i in observation_indices],
            # actions
            self.task_cfg.action_key: [i / self.task_cfg.fps for i in action_indices],
        }
    
    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        obs_queue_size = (self.cfg.n_obs_steps-1)*self.cfg.obs_step_size+1
        self._queues = {
            self.task_cfg.action_key: deque(maxlen=self.cfg.n_action_steps),
            self.task_cfg.state_key: deque(maxlen=obs_queue_size),
            **{key: deque(maxlen=obs_queue_size) for key in self.task_cfg.image_keys},
        }
        self.temporal_ensembler.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], return_viz: bool = False) -> Tensor:
        self.eval()
        assert self.task_cfg.state_key in batch, f"Missing {self.task_cfg.state_key} in batch"
        assert all(key in batch for key in self.task_cfg.image_keys), f"Missing {self.task_cfg.image_keys} in batch"
        self._queues = populate_queues(self._queues, batch)

        viz = {}
        if len(self._queues[self.task_cfg.action_key]) == 0:
            batch = {
                k: torch.stack(list(self._queues[k])[::self.cfg.obs_step_size], dim=1) 
                for k in self.task_cfg.image_keys + [self.task_cfg.state_key]
            }
            batch = self.normalize_inputs(batch)
            actions, viz = self.flow.generate_actions(batch)
            actions = self.unnormalize_outputs({self.task_cfg.action_key: actions})[self.task_cfg.action_key]
            actions = self.temporal_ensembler.update(
                actions[:, :, :self.task_cfg.action_dim],
                n=self.cfg.n_action_steps
            )
            # actions = actions[:, :self.cfg.n_action_steps, :self.task_cfg.action_dim]
            self._queues[self.task_cfg.action_key].extend(actions.transpose(0, 1))

        action = self._queues[self.task_cfg.action_key].popleft()        
        if return_viz: return action, viz
        return action

    @torch.no_grad()
    def get_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)
        actions, viz = self.flow.generate_actions(batch)
        actions = self.unnormalize_outputs({self.task_cfg.action_key: actions})[self.task_cfg.action_key]
        return actions, viz

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, loss_dict = self.flow.compute_loss(batch)
        return loss, loss_dict
    
def _make_flow_matcher(name: str, **kwargs: dict) -> cfm.ConditionalFlowMatcher:
    if name == "conditional":
        return cfm.ConditionalFlowMatcher(**kwargs)
    elif name == "target":
        return cfm.TargetConditionalFlowMatcher(**kwargs)
    else:
        raise ValueError(f"Unsupported flow matcher type {name}")
        
class FlowModel(nn.Module):
    def __init__(self, policy_cfg: FlowPolicyConfig, task_cfg: TaskConfig):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg

        self.flow_matcher = _make_flow_matcher(
            policy_cfg.flow_matcher,
            **policy_cfg.flow_matcher_kwargs,
        )

        # observation processing
        self.resize = torchvision.transforms.Resize(policy_cfg.resize_shape)
        vision_encoders = {}
        self.image_encoder_keys = [k.split(".")[-1] for k in task_cfg.image_keys]
        for key in self.image_encoder_keys:
            vision_encoders[key] = VisionEncoder(
                resize_shape=policy_cfg.peripheral_shape,
                crop_shape=policy_cfg.peripheral_crop,
                crop_is_random=True,
                use_spatial_softmax=policy_cfg.use_spatial_softmax,
                num_kp=policy_cfg.num_kp,
                out_dim=policy_cfg.dim_model,
            )
        self.vision_encoders = nn.ModuleDict(vision_encoders)

        self.cond_dim = len(self.image_encoder_keys) * 512 + task_cfg.state_dim
        
        self.unet = DiffusionConditionalUnet1d(
            global_cond_dim=self.cond_dim,
            action_dim=task_cfg.action_dim,
        )

    def prepare_global_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        viz = {}
        cond = []

        cond.append(
            batch[self.task_cfg.state_key].flatten(start_dim=1)
        ) 

        # process images
        for image_key, image_encoder_key in zip(self.task_cfg.image_keys, self.image_encoder_keys):
            images = batch[image_key]  # (B, S, C, H, W)
            batch_size = images.shape[0]
            images = einops.rearrange(images, 'b s c h w -> (b s) c h w')
            images = self.resize(images)  # (B, S, C, H, W)
            feat = self.vision_encoders[image_encoder_key](images)  
            feat = einops.rearrange(feat, '(b s) d -> b (s d)', b=batch_size)  # (B, S, D)
            cond.append(feat)  # (B, S, D)
            if not self.training:
                viz[image_encoder_key] = images

        cond = torch.cat(cond, dim=1)

        return cond, viz

    def predict(self, noise: Tensor, timestep: Tensor, global_cond: dict) -> Tensor:
        x = self.unet.forward(
            x=noise,
            timestep=timestep,
            global_cond=global_cond,
        )
        return x

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        num_steps = self.cfg.n_sampling_steps
        shape = (batch_size, self.cfg.horizon, self.task_cfg.action_dim)
        x = torch.randn(shape, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        for n in range(num_steps):
            timestep = n * dt * torch.ones(x.shape[0], device=x.device)
            vt = self.predict(x, timestep, global_cond)
            x = x + vt * dt

        return x

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[self.task_cfg.state_key].shape[0]
        global_cond, viz = self.prepare_global_conditioning(batch) 
        actions = self.conditional_sample(batch_size, global_cond=global_cond)
        return actions, viz

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond, _ = self.prepare_global_conditioning(batch) 
        x0 = torch.randn_like(batch[self.task_cfg.action_key]) 
        timestep, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, batch[self.task_cfg.action_key])
        vt = self.predict(xt, timestep, global_cond=global_cond)
        flow_matching_loss = F.mse_loss(vt, ut, reduction="none").mean()
        loss = flow_matching_loss
        loss_dict = {
            "flow_matching_loss": flow_matching_loss.item(),
        }
        return loss, loss_dict

class VisionEncoder(nn.Module):
    def __init__(
        self, 
        resize_shape: tuple[int, int],
        crop_shape: tuple[int, int] | None = None,
        crop_is_random: bool = True,
        num_kp: int = 32,
        out_dim: int = 512,
        use_spatial_softmax: bool = False,
    ):
        super().__init__()
        # pre proc
        self.resize = torchvision.transforms.Resize(resize_shape)
        if crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})

        # post proc
        if not self.do_crop:
            dummy_shape = (1, 3, *resize_shape)
        else:
            dummy_shape = (1, 3, *crop_shape)
        feature_map_shape = get_output_shape(
            lambda x: self.backbone(x)["feature_map"], dummy_shape
        )[1:]

        if use_spatial_softmax:
            self.pool = SpatialSoftmax(feature_map_shape, num_kp=num_kp)
            self.out = nn.Sequential(
                nn.Linear(num_kp*2, num_kp*2),
                nn.ReLU(inplace=True),
                nn.Linear(num_kp*2, out_dim),
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out = nn.Linear(feature_map_shape[0], out_dim) if feature_map_shape[0] != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.resize(x)
        if self.do_crop:
            if self.training:  
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = self.backbone(x)["feature_map"]
        x = self.pool(x) # (B, K, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
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
    
def crop_at_center(images: Tensor, centers: Tensor, crop_shape: tuple):
    # crop the images around the predicted gaze
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
        output_size=crop_shape,
    )   
    return images

class TemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.
        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor, n: int  = 1) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        assert n <= self.chunk_size, "n must be less than or equal to chunk_size."
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-n] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -n:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-n:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, :n],
            self.ensembled_actions[:, n:],
            self.ensembled_actions_count[n:],
        )
        return action