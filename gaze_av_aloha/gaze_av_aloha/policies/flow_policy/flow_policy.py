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
    populate_queues,
)
import torch.nn.functional as F
import einops
from gaze_av_aloha.policies.flow_policy.transformer import DiT, AttentionPooling, get_1d_rotary_embed, get_nd_rotary_embed
from gaze_av_aloha.policies.flow_policy.gaze import gaze_crop, gaze_mask
from gaze_av_aloha.policies.flow_policy.dino import DINO
from torchvision.transforms import Resize
import logging
import numpy as np

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
                **{k: policy_cfg.image_norm_mode for k in policy_cfg.image_to_gaze_key.keys()},
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

        if policy_cfg.use_temporal_ensemble:
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
        backbone_condition = lambda n: "flow.dino" in n
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
        if self.cfg.use_ema:
            logging.info(f"[FlowPolicy] Initializing EMA with decay {self.cfg.ema_decay}")
            return EMAModel(
                parameters=self.parameters(),
                decay=self.cfg.ema_decay,
            ) 
        else:
            logging.info(f"[FlowPolicy] EMA is not used.")
            return None
    
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
                for k in self.cfg.image_to_gaze_key.keys()
            },
            self.task_cfg.state_key: [i / self.task_cfg.fps for i in observation_indices],
            # gaze
            **{
                k: [i / self.task_cfg.fps for i in observation_indices]
                for k in self.cfg.image_to_gaze_key.values()
            },
            # actions
            self.task_cfg.action_key: [i / self.task_cfg.fps for i in action_indices],
        }
    
    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        obs_queue_size = (self.cfg.n_obs_steps-1)*self.cfg.obs_step_size+1
        self._queues = {
            self.task_cfg.action_key: deque(maxlen=self.cfg.n_action_steps),
            self.task_cfg.state_key: deque(maxlen=obs_queue_size),
            **{key: deque(maxlen=obs_queue_size) for key in self.cfg.image_to_gaze_key.keys()},
        }
        if self.cfg.use_temporal_ensemble:
            self.temporal_ensembler.reset()   

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], return_viz: bool = False) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch)

        viz = {}
        if len(self._queues[self.task_cfg.action_key]) == 0:
            batch = {
                k: torch.stack(list(self._queues[k])[::self.cfg.obs_step_size], dim=1) 
                for k in list(self.cfg.image_to_gaze_key.keys()) + [self.task_cfg.state_key]
            }
            batch = self.normalize_inputs(batch)
            actions, viz = self.flow.generate_actions(batch)
            actions = self.unnormalize_outputs({self.task_cfg.action_key: actions})[self.task_cfg.action_key]

            if self.cfg.use_temporal_ensemble:
                actions = self.temporal_ensembler.update(actions, n=self.cfg.n_action_steps)
            else:
                actions = actions[:, :self.cfg.n_action_steps]

            self._queues[self.task_cfg.action_key].extend(actions.transpose(0, 1))

        action = self._queues[self.task_cfg.action_key].popleft()

        if return_viz: return action, viz
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, loss_dict = self.flow.compute_loss(batch)
        return loss, loss_dict
       
class FlowModel(nn.Module):
    def __init__(self, policy_cfg: FlowPolicyConfig, task_cfg: TaskConfig):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg

        # img encoding
        self.dino = DINO(num_freeze_layers=policy_cfg.dino_freeze_n_layers)
        self.dino_proj = nn.Linear(self.dino.embed_dim, policy_cfg.dim_model)
        assert policy_cfg.crop_shape[0] % self.dino.patch_size == 0 and \
               policy_cfg.crop_shape[1] % self.dino.patch_size == 0, \
            f"Crop shape {policy_cfg.crop_shape} must be divisible by dino patch size {self.dino.patch_size}."
        self.patch_shape = (policy_cfg.crop_shape[0] // self.dino.patch_size,
                            policy_cfg.crop_shape[1] // self.dino.patch_size)

        # state encoding
        self.state_proj = nn.Sequential(
            nn.Dropout(policy_cfg.state_dropout),
            nn.Linear(task_cfg.state_dim, policy_cfg.dim_model),
            nn.GELU(),
            nn.Dropout(policy_cfg.state_dropout),
            nn.Linear(policy_cfg.dim_model, policy_cfg.dim_model),
            nn.LayerNorm(policy_cfg.dim_model),
        )

        # attention pooling
        self.pool = AttentionPooling(
            hidden_size=policy_cfg.dim_model,
            num_queries=policy_cfg.pool_n_queries,
            depth=policy_cfg.pool_n_layers,
            num_heads=policy_cfg.n_heads,
            mlp_ratio=policy_cfg.mlp_ratio,
            dropout=policy_cfg.dropout,
        ) 
        
        # diffusion transformer
        self.DiT = DiT(
            in_dim=task_cfg.action_dim,
            out_dim=task_cfg.action_dim,
            hidden_size=policy_cfg.dim_model,
            depth=policy_cfg.dit_n_layers,
            num_heads=policy_cfg.n_heads,
            mlp_ratio=policy_cfg.mlp_ratio,
            dropout=policy_cfg.dropout,
            time_dim=policy_cfg.time_dim,
        )

        s = policy_cfg.n_obs_steps
        n = len(self.cfg.image_to_gaze_key)
        h, w = self.patch_shape
        self.register_buffer(
            'attn_pooling_pos_embed',
            torch.cat([
                get_1d_rotary_embed(
                    dim=policy_cfg.dim_model // policy_cfg.n_heads,
                    pos=torch.arange(s * (1 + n*(1 + self.dino.num_register_tokens)), dtype=torch.float32),
                ),
                get_nd_rotary_embed(
                    dim=policy_cfg.dim_model // policy_cfg.n_heads,
                    grid_shape=(s, n, h, w)
                ),
            ], dim=0)
        )
        self.register_buffer(
            'dit_x_pos_embed',
            get_1d_rotary_embed(
                dim=policy_cfg.dim_model // policy_cfg.n_heads,
                pos=torch.arange(policy_cfg.horizon, dtype=torch.float32),
            )
        )
        self.register_buffer(
            'dit_c_pos_embed',
            get_1d_rotary_embed(
                dim=policy_cfg.dim_model // policy_cfg.n_heads,
                pos=torch.arange(policy_cfg.pool_n_queries, dtype=torch.float32),
            )
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

        img = torch.stack([batch[key] for key in self.cfg.image_to_gaze_key.keys()], dim=2)
        b, s, n = img.shape[:3]
        img = einops.rearrange(img, 'b s n c h w -> (b s n) c h w')  # (B*S*N, C, H, W)
        img = Resize(self.cfg.resize_shape)(img)  # (B, S, C, H, W)

        if self.training and torch.rand(1) < self.cfg.gaze_prob:
            gaze = torch.stack([batch[key] for key in self.cfg.image_to_gaze_key.values()], dim=2)
            gaze = einops.rearrange(gaze, 'b s n c -> (b s n) c')  # (B*S*N, C)
            img, gaze = gaze_crop(images=img, crop_shape=self.cfg.crop_shape, gaze=gaze, crop_is_random=self.cfg.crop_is_random)
            masks = gaze_mask(gaze=gaze, patch_shape=self.patch_shape, sigma=self.cfg.gaze_sigma, k=self.cfg.gaze_k, random=False)
        else:
            img, _ = gaze_crop(images=img, crop_shape=self.cfg.crop_shape, crop_is_random=self.training and self.cfg.crop_is_random)
            masks = None
        dino_feat = self.dino(img, masks=masks)  # (B*S, L, D)

        tokens = []
        tokens.append(self.state_proj(batch[self.task_cfg.state_key])) # (B, S, D)
        tokens.append(
            einops.rearrange(
                self.dino_proj(dino_feat['x_norm_clstoken']),
                '(b s n) d -> b (s n) d',
                b=b, s=s, n=n
            )  
        )
        tokens.append(
            einops.rearrange(
                self.dino_proj(dino_feat['x_norm_regtokens']),
                '(b s n) r d -> b (s n r) d',
                b=b, s=s, n=n, r=self.dino.num_register_tokens
            )
        )
        tokens.append(
            einops.rearrange(
                self.dino_proj(dino_feat['x_norm_patchtokens']),
                '(b s n) l d -> b (s n l) d',
                b=b, s=s, n=n, l=self.patch_shape[0] * self.patch_shape[1]
            )
        )
        tokens = torch.cat(tokens, dim=1) 
        cond = self.pool(
            c=tokens,
            c_pos_emb=self.attn_pooling_pos_embed,
        )

        if not self.training:
            viz['input'] = img

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
            vt = self.DiT(x=xt, c=global_cond, t=time.expand(batch_size), x_pos_emb=self.dit_x_pos_embed, c_pos_emb=self.dit_c_pos_embed)
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

        vt = self.DiT(x=xt, c=global_cond, t=time, x_pos_emb=self.dit_x_pos_embed, c_pos_emb=self.dit_c_pos_embed)
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