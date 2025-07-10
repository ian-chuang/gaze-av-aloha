import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import math

from diffusers.models.embeddings import get_timestep_embedding

from gaze_av_aloha.policies.cage_policy.cage_policy_config import CAGEPolicyConfig
from gaze_av_aloha.configs import TaskConfig
from gaze_av_aloha.policies.policy import Policy

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from gaze_av_aloha.policies.normalize import Normalize, Unnormalize
from torchvision.transforms import Resize
from gaze_av_aloha.utils.policy_utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from collections import deque

from gaze_av_aloha.policies.cage_policy.vision import get_vision_encoder
# from gaze_av_aloha.policies.cage_policy.unet import ConditionalAttnUnet1D
from gaze_av_aloha.policies.cage_policy.dit import DiT
from gaze_av_aloha.policies.cage_policy.pool import AttentionPooling

from torchcfm import ConditionalFlowMatcher

import logging

class CAGEPolicy(Policy):
    def __init__(
        self, 
        policy_cfg: CAGEPolicyConfig,
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
        backbone_condition = lambda n: "flow.backbone." in n
        if not any(backbone_condition(n) for n, p in self.named_parameters()):
            raise ValueError(f"No parameters found satifying the condition for vision backbone: {backbone_condition}")
        
        # TODO REMOVE THIS LOGGING
        for n, p in self.named_parameters():
            if backbone_condition(n) and p.requires_grad:
                logging.info(f"{n}: {self.cfg.optimizer_lr_backbone}.")

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
        # indices
        observation_indices = list(reversed(range(0, -self.cfg.n_obs_steps*self.cfg.obs_step_size, -self.cfg.obs_step_size)))
        action_indices = [i - 1 for i in observation_indices] + list(range(self.cfg.horizon))
        gaze_indices = observation_indices + list(range(1, self.cfg.horizon+1))
        assert len(observation_indices) == self.cfg.n_obs_steps, "Observation indices length mismatch"

        # timestamps
        observation_timestamps = [i / self.task_cfg.fps for i in observation_indices]
        action_timestamps = [i / self.task_cfg.fps for i in action_indices]
        gaze_timestamps = [i / self.task_cfg.fps for i in gaze_indices]

        return {
            **{k: observation_timestamps for k in self.cfg.image_to_gaze_key.keys()}, # images
            **{k: gaze_timestamps for k in self.cfg.image_to_gaze_key.values()}, # gaze
            self.task_cfg.state_key: observation_timestamps, # state
            self.task_cfg.action_key: action_timestamps, # action
        }
    
    def reset(self):
        obs_queue_size = (self.cfg.n_obs_steps-1)*self.cfg.obs_step_size+1
        self._obs_queue = {
            **{key: deque(maxlen=obs_queue_size) for key in self.cfg.image_to_gaze_key.keys()},
            **{key: deque(maxlen=obs_queue_size) for key in self.cfg.image_to_gaze_key.values()}, # gaze
            self.task_cfg.state_key: deque(maxlen=obs_queue_size),
            self.task_cfg.action_key: deque(maxlen=obs_queue_size),
        }
        self._act_queues = {
            **{key: deque(maxlen=self.cfg.n_action_steps) for key in self.cfg.image_to_gaze_key.values()}, # gaze
            self.task_cfg.action_key: deque(maxlen=self.cfg.n_action_steps),
        }
        if self.cfg.use_temporal_ensemble:
            self.temporal_ensembler.reset()   

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], return_viz: bool = False) -> Tensor:
        self.eval()
        self._obs_queue = populate_queues(self._obs_queue, batch)

        if len(self._obs_queue[self.task_cfg.action_key]) == 0:
            state = batch[self.task_cfg.state_key]
            state_queue = self._obs_queue[self.task_cfg.state_key]
            maxlen = state_queue.maxlen
            device = state.device
            dtype = state.dtype
            batch_size = state.shape[0]
            self._obs_queue[self.task_cfg.action_key].extend([state] * maxlen)
            gaze_padding = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            for gaze_key in self.cfg.image_to_gaze_key.values():
                self._obs_queue[gaze_key].extend([gaze_padding] * maxlen)

        viz = {}
        if len(self._act_queues[self.task_cfg.action_key]) == 0:
            task = batch.get("task", None)
            batch = {
                k: torch.stack(list(v)[::self.cfg.obs_step_size], dim=1) 
                for k, v in self._obs_queue.items()
            }
            if task: batch["task"] = task
            batch = self.normalize_inputs(batch)
            outputs, viz = self.flow.generate_actions(batch)
            outputs = self.unnormalize_outputs(outputs)

            if self.cfg.use_temporal_ensemble:
                outputs = self.temporal_ensembler.update(outputs, n=self.cfg.n_action_steps)
            else:
                outputs = {k: v[:, :self.cfg.n_action_steps] for k, v in outputs.items()}

            for key in outputs:
                self._act_queues[key].extend(outputs[key].transpose(0, 1))

        for key in self._act_queues:
            if len(self._act_queues[key]) > 0:
                item = self._act_queues[key].popleft()
                self._obs_queue[key].append(item)
        action = self._obs_queue[self.task_cfg.action_key][-1]  # Get the last action from the queue

        if return_viz: return action, viz
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)

        # pad gaze obs with zeros
        for gaze_key in self.cfg.image_to_gaze_key.values():
            is_pad = batch[self.task_cfg.action_key + "_is_pad"][:, :self.cfg.n_obs_steps] # use action key since it is 1 behind
            batch[gaze_key][:, :self.cfg.n_obs_steps][is_pad] = 0.0
        
        # pad action obs with state TODO not sure this is correct
        is_pad = batch[self.task_cfg.action_key + "_is_pad"][:, :self.cfg.n_obs_steps]
        state_pad = batch[self.task_cfg.state_key][:, :self.cfg.n_obs_steps][is_pad]
        batch[self.task_cfg.action_key][:, :self.cfg.n_obs_steps][is_pad] = state_pad

        loss, loss_dict = self.flow.compute_loss(batch)
        return loss, loss_dict
       
class FlowModel(nn.Module):
    def __init__(self, policy_cfg: CAGEPolicyConfig, task_cfg: TaskConfig):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg

        self.target_keys_to_dim = {
            task_cfg.action_key: task_cfg.action_dim,
            **{k: 2 for k in policy_cfg.image_to_gaze_key.values() if policy_cfg.use_gaze},
        }
        self.noise_dim = sum(self.target_keys_to_dim.values())

        self.noise_proj = nn.Linear(self.noise_dim, policy_cfg.dim_model)
        self.state_proj = nn.Sequential(
            nn.Dropout(policy_cfg.dropout),
            nn.Linear(task_cfg.state_dim, policy_cfg.dim_model),
        )
        self.gaze_proj = nn.Sequential(
            nn.Dropout(policy_cfg.dropout),
            nn.Linear(2, policy_cfg.dim_model),
        ) if policy_cfg.use_gaze else None
        self.backbone = get_vision_encoder(
            name=policy_cfg.vision_encoder,
            **policy_cfg.vision_encoder_kwargs,
        )
        self.backbone_proj = nn.Linear(self.backbone.embed_dim, policy_cfg.dim_model)
        self.pool = AttentionPooling(
            num_queries=policy_cfg.pool_n_queries,
            hidden_size=policy_cfg.dim_model,
            depth=policy_cfg.pool_n_layers,
            num_heads=policy_cfg.n_heads,
            dropout=policy_cfg.dropout,
        )
        self.dit = DiT(
            dim=policy_cfg.dim_model,
            out_dim=self.noise_dim,
            depth=policy_cfg.dit_n_layers,
            num_heads=policy_cfg.n_heads,
            dropout=policy_cfg.dropout,
            time_dim=policy_cfg.dit_time_dim,
        )

        self.obs_step_pos_emb = nn.Parameter(torch.randn(policy_cfg.n_obs_steps, policy_cfg.dim_model))
        self.image_pos_emb = nn.Parameter(torch.randn(len(policy_cfg.image_to_gaze_key), policy_cfg.dim_model))
        self.dit_pos_emb = nn.Parameter(torch.randn(policy_cfg.horizon, policy_cfg.dim_model))
        
    def prepare_global_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        tokens = []
        pos_emb = []

        img = torch.stack([batch[key] for key in self.cfg.image_to_gaze_key.keys()], dim=2)
        b, s, n = img.shape[:3]
        img = einops.rearrange(img, 'b s n c h w -> (b s n) c h w')  
        img = Resize(self.cfg.input_shape)(img)  

        tokens.append(self.state_proj(batch[self.task_cfg.state_key])) # (B, S, D)
        pos_emb.append(self.obs_step_pos_emb.unsqueeze(0))

        if self.cfg.use_gaze:
            gaze = torch.stack([batch[key] for key in self.cfg.image_to_gaze_key.values()], dim=2)
            gaze = einops.rearrange(gaze, 'b s n d -> (b s n) d') 
            if self.training:
                gaze = gaze + torch.randn_like(gaze) * self.cfg.gaze_noise

            tokens.append(
                einops.rearrange(
                    self.gaze_proj(gaze), 
                    '(b s n) d -> b (s n) d',
                    b=b, s=s, n=n
                )
            )
            pos_emb.append(
                einops.rearrange(
                    self.obs_step_pos_emb.unsqueeze(1) + self.image_pos_emb.unsqueeze(0),
                    's n d -> 1 (s n) d', 
                    s=s, n=n
                )
            )
        else:
            gaze = None

        img_tokens, viz = self.backbone(img, centers=gaze)
        tokens.append(
            einops.rearrange(
                self.backbone_proj(img_tokens),
                '(b s n) l d -> b (s n l) d',
                b=b, s=s, n=n,
            )
        )
        pos_emb.append(
            einops.repeat(
                self.obs_step_pos_emb.unsqueeze(1) + self.image_pos_emb.unsqueeze(0),
                's n d -> 1 (s n l) d',
                s=s, n=n, l=img_tokens.shape[1]
            )
        )

        tokens = torch.cat(tokens, dim=1)  # (B, L, D)
        pos_emb = torch.cat(pos_emb, dim=1)  # (1, L, D)
        cond = self.pool(tokens, pos_emb)

        return cond, viz
    
    def predict_velocity(self, noise: Tensor, global_cond: dict, timestep: Tensor) -> Tensor:
        vt = self.dit.forward(
            sample=self.noise_proj(noise) + self.dit_pos_emb.unsqueeze(0),
            condition=global_cond,
            timestep=timestep, 
        )
        return vt
    
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

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)

        actions_shape = (batch_size, self.cfg.horizon, self.noise_dim)
        noise = self.sample_noise(actions_shape, device=device)

        dt = -1.0 / self.cfg.n_sampling_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        xt = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            vt = self.predict_velocity(noise=xt, global_cond=global_cond, timestep=time.expand(batch_size))
            # Euler step
            xt += dt * vt
            time += dt
        return xt

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[self.task_cfg.state_key].shape[0]
        global_cond, viz = self.prepare_global_conditioning(batch) 
        sample = self.conditional_sample(batch_size, global_cond=global_cond)
        outputs = {
            k: v
            for k, v in zip(
                self.target_keys_to_dim.keys(),
                torch.split(sample, list(self.target_keys_to_dim.values()), dim=-1),
            )
        }
        return outputs, viz

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        batch = dict(batch)  

        x1 = []
        for key in self.target_keys_to_dim:
            x1.append(batch[key][:, self.cfg.n_obs_steps:])
            batch[key] = batch[key][:, :self.cfg.n_obs_steps]
        x1 = torch.cat(x1, dim=-1)

        global_cond, _ = self.prepare_global_conditioning(batch) 

        noise = self.sample_noise(x1.shape, x1.device)
        time = self.sample_time(x1.shape[0], x1.device)
        time_expanded = time[:, None, None]
        xt = time_expanded * noise + (1 - time_expanded) * x1
        ut = noise - x1

        vt = self.predict_velocity(noise=xt, global_cond=global_cond, timestep=time)
        flow_matching_loss = F.mse_loss(vt, ut, reduction="none").mean()
        loss = flow_matching_loss
        loss_dict = {
            "flow_matching_loss": flow_matching_loss.item(),
        }
        return loss, loss_dict
    
def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

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
        self.keys_to_dim = None
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, outputs: dict[str, Tensor], n: int  = 1) -> Tensor:
        """
        Args:
            outputs (Tensor): A dictionary of actions
        """
        if self.keys_to_dim is None: self.keys_to_dim = {k: v.shape[-1] for k, v in outputs.items()}
        actions = torch.cat([outputs[k] for k in self.keys_to_dim], dim=-1)

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

        return {
            k: v 
            for k, v in zip(
                self.keys_to_dim.keys(), 
                torch.split(action, list(self.keys_to_dim.values()), dim=-1)
            )
        }