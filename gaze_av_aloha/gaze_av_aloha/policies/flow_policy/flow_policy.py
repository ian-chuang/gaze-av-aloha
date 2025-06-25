from torch import nn, Tensor
from gaze_av_aloha.policies.flow_policy.flow_policy_config import FlowPolicyConfig
from gaze_av_aloha.configs import TaskConfig
from gaze_av_aloha.policies.policy import Policy
import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from gaze_av_aloha.policies.normalize import Normalize, Unnormalize
from collections import deque
from gaze_av_aloha.utils.policy_utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
import torchcfm.conditional_flow_matching as cfm
import torch.nn.functional as F
import einops
from gaze_av_aloha.policies.flow_policy.transformer import DiT, AttentionPooling
from gaze_av_aloha.policies.flow_policy.dino import DINO
from torchvision.transforms import Resize
import logging
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


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
            temporal_ensemble_coeff=policy_cfg.temporal_ensemble_coeff,
            chunk_size=policy_cfg.horizon,
        )
        self.reset()

    def get_optimizer(self) -> torch.optim.Optimizer:
        logging.info(f"""
            [FlowPolicy] Initializing AdamW optimizer with the following parameters:
            - Learning Rate: {self.cfg.optimizer_lr}
            - Learning Rate for Backbone: {self.cfg.optimizer_lr_backbone}
            - Betas: {self.cfg.optimizer_betas}
            - Epsilon: {self.cfg.optimizer_eps}
            - Weight Decay: {self.cfg.optimizer_weight_decay}
        """)
        backbone_condition = lambda n: "backbone" in n
        if not any(backbone_condition(n) for n, p in self.named_parameters()):
            raise ValueError(f"No parameters found satifying the condition for vision backbone: {backbone_condition}")
        
        for n, p in self.named_parameters():
            if backbone_condition(n):
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
        logging.info(f"""
            [FlowPolicy] Initializing scheduler '{self.cfg.scheduler_name}' with the following parameters:
            - Warmup Steps: {self.cfg.scheduler_warmup_steps}
            - Total Training Steps: {num_training_steps}
        """)
        return get_scheduler(
            name=self.cfg.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def get_ema(self) -> EMAModel | None:
        logging.info(f"[FlowPolicy] Initializing EMA with decay {self.cfg.ema_decay} and use_ema={self.cfg.use_ema}")
        return EMAModel(
            parameters=self.parameters(),
            decay=self.cfg.ema_decay,
        ) if self.cfg.use_ema else None
    
    def get_delta_timestamps(self):
        observation_indices = list(reversed(range(0, -self.cfg.n_obs_steps*self.cfg.obs_step_size, -self.cfg.obs_step_size)))
        assert len(observation_indices) == self.cfg.n_obs_steps, "Observation indices length mismatch"
        action_indices = list(range(self.cfg.horizon))
        logging.info(f"""
            [FlowPolicy] Delta Timestamps:
            Observation Indices: {observation_indices}
            Action Indices: {action_indices}
        """)
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
            actions = self.temporal_ensembler.update(actions, n=self.cfg.n_action_steps)
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

        # observation processing
        self.resize = Resize(policy_cfg.resize_shape)
        self.input_resizer = Resize(policy_cfg.input_shape)
        self.backbone = DINO(policy_cfg.dino_freeze_n_layers)
        self.backbone_proj = nn.Linear(self.backbone.embed_dim, policy_cfg.dim_model)
        self.pool = AttentionPooling(
            hidden_size=policy_cfg.dim_model,
            out_dim=policy_cfg.pool_out_dim,
            num_queries=policy_cfg.pool_n_queries,
            depth=policy_cfg.pool_n_layers,
            num_heads=policy_cfg.n_heads,
            mlp_ratio=policy_cfg.mlp_ratio,
            dropout=policy_cfg.dropout,
        ) 
        # backbone = torchvision.models.resnet18(
        #     weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        #     norm_layer=FrozenBatchNorm2d,
        # )
        # self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.state_dropout = nn.Dropout(policy_cfg.state_dropout)
        # self.state_embed = nn.Sequential(
        #     nn.Dropout(policy_cfg.state_dropout),
        #     nn.Linear(task_cfg.state_dim, policy_cfg.dim_model),
        #     nn.GELU(),
        #     nn.Dropout(policy_cfg.state_dropout),
        #     nn.Linear(policy_cfg.dim_model, policy_cfg.dim_model),
        # )
        

        # diffusion transformer
        self.DiT = DiT(
            in_dim=task_cfg.action_dim,
            out_dim=task_cfg.action_dim,
            cond_dim=policy_cfg.pool_out_dim + task_cfg.state_dim * policy_cfg.n_obs_steps,  
            hidden_size=policy_cfg.dim_model,
            depth=policy_cfg.dit_n_layers,
            num_heads=policy_cfg.n_heads,
            mlp_ratio=policy_cfg.mlp_ratio,
            dropout=policy_cfg.dropout,
            time_dim=policy_cfg.time_dim,
        )

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def prepare_global_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        viz = {}
        batch = dict(batch)  # make a copy to avoid modifying the original batch

        cond = []
        for image_key in self.task_cfg.image_keys:
            img = batch[image_key]  # (B, S, C, H, W)
            batch_size = img.shape[0]
            img = einops.rearrange(img, 'b s c h w -> (b s) c h w')
            img = self.resize(img) 
            img = self.input_resizer(img)  # (B*S, C, H, W)
            feat = self.backbone_proj(self.backbone(img))
            feat = self.pool(feat)
            feat = einops.rearrange(feat, '(b s) d -> b (s d)', b=batch_size)  # (B, S*C)
            cond.append(feat)  # (B*S, C)
            if not self.training:
                viz[image_key] = img
        cond.append(self.state_dropout(batch[self.task_cfg.state_key]).flatten(start_dim=1))  
        cond = torch.cat(cond, dim=1) 


        # tokens = [self.state_embed(batch[self.task_cfg.state_key])]  # (B, S, D)
        # for image_key in self.task_cfg.image_keys:
        #     img = batch[image_key]  # (B, S, C, H, W)
        #     batch_size = img.shape[0]
        #     img = einops.rearrange(img, 'b s c h w -> (b s) c h w')
        #     img = self.resize(img) 
        #     dino_feat = self.dino(img)
        #     dino_feat = einops.rearrange(dino_feat, '(b s) l d -> b (s l) d', b=batch_size)
        #     tokens.append(self.dino_embed(dino_feat))  # (B, S*L, D)
        #     if not self.training:
        #         viz[image_key] = img

        # tokens = torch.cat(tokens, dim=1)  
        # cond = self.attn_pooling(tokens) 
        return cond, viz

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)

        actions_shape = (batch_size, self.cfg.horizon, self.task_cfg.action_dim)
        noise = self.sample_noise(actions_shape, device=device)

        dt = -1.0 / self.cfg.n_sampling_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        xt = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            vt = self.DiT(x=xt, timestep=time.expand(batch_size), cond=global_cond)
            # Euler step
            xt += dt * vt
            time += dt
        return xt

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[self.task_cfg.state_key].shape[0]
        global_cond, viz = self.prepare_global_conditioning(batch) 
        actions = self.conditional_sample(batch_size, global_cond=global_cond)
        return actions, viz

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond, _ = self.prepare_global_conditioning(batch) 

        actions = batch[self.task_cfg.action_key]
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        xt = time_expanded * noise + (1 - time_expanded) * actions
        ut = noise - actions

        vt = self.DiT(x=xt, timestep=time, cond=global_cond)
        flow_matching_loss = F.mse_loss(vt, ut, reduction="none").mean()
        loss = flow_matching_loss
        loss_dict = {
            "flow_matching_loss": flow_matching_loss.item(),
        }
        return loss, loss_dict

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