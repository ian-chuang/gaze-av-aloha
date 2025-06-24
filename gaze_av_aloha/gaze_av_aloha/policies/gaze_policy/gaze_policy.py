from torch import nn, Tensor
from gaze_av_aloha.policies.gaze_policy.gaze_policy_config import GazePolicyConfig
from gaze_av_aloha.configs import TaskConfig
from gaze_av_aloha.policies.policy import Policy
import torch
from diffusers.optimization import get_scheduler
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
from gaze_av_aloha.policies.gaze_policy.vision import VisionEncoder, crop_at_center
from gaze_av_aloha.policies.gaze_policy.transformer import DiT, AttentionPooling, get_foveated_pos_embed, get_1d_rotary_embed, TransformerEncoder
from torchvision.transforms import Resize
import logging
from torchvision.ops import roi_align

class GazePolicy(Policy):
    def __init__(
        self, 
        policy_cfg: GazePolicyConfig,
        task_cfg: TaskConfig,
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg
        self.stats = stats

        # hacky add eye mean std to stats of action (eye is already normalized to -1, 1)
        self.action_dim = task_cfg.action_dim + len(policy_cfg.image_to_gaze_key)*2
        if len(stats[task_cfg.action_key]["mean"]) == task_cfg.action_dim:
            stats[task_cfg.action_key]["mean"].extend([0] * len(policy_cfg.image_to_gaze_key) * 2)
            stats[task_cfg.action_key]["std"].extend([1] * len(policy_cfg.image_to_gaze_key) * 2)
            stats[task_cfg.action_key]["min"].extend([-1] * len(policy_cfg.image_to_gaze_key) * 2)
            stats[task_cfg.action_key]["max"].extend([1] * len(policy_cfg.image_to_gaze_key) * 2)
            stats["action_history"] = stats[task_cfg.action_key]

        assert all(
            key in task_cfg.image_keys for key in policy_cfg.image_to_gaze_key
        ), f"All keys in image_to_gaze_key must be in task_cfg.image_keys. Found: {policy_cfg.image_to_gaze_key.keys()} not in {task_cfg.image_keys}"

        self.normalize_inputs = Normalize(
            {
                **{k: policy_cfg.image_norm_mode for k in task_cfg.image_keys},
                task_cfg.state_key: policy_cfg.state_norm_mode,
                "action_history": policy_cfg.action_norm_mode,
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

        padding = policy_cfg.action_history_padding + [0] * len(policy_cfg.image_to_gaze_key) * 2
        assert len(padding) == self.action_dim
        self.register_buffer(
            "action_history_padding",
            torch.tensor(padding)
        )

        self.action_temporal_ensembler = TemporalEnsembler(
            temporal_ensemble_coeff=policy_cfg.action_temporal_ensemble_coeff,
            chunk_size=policy_cfg.horizon,
        )
        self.gaze_temporal_ensembler = TemporalEnsembler(
            temporal_ensemble_coeff=policy_cfg.gaze_temporal_ensemble_coeff,
            chunk_size=policy_cfg.horizon,
        )

        self.reset()

    def get_optimizer(self) -> torch.optim.Optimizer:
        logging.info(f"""
            [GazePolicy] Initializing AdamW optimizer with the following parameters:
            - Learning Rate: {self.cfg.optimizer_lr}
            - Learning Rate for Backbone: {self.cfg.optimizer_lr_backbone}
            - Betas: {self.cfg.optimizer_betas}
            - Epsilon: {self.cfg.optimizer_eps}
            - Weight Decay: {self.cfg.optimizer_weight_decay}
        """)
        backbone_condition = lambda n: "flow" in n and "dino" in n
        if not any(backbone_condition(n) for n, p in self.named_parameters()):
            raise ValueError(f"No parameters found satifying the condition for vision backbone: {backbone_condition}")
        
        for n, p in self.named_parameters():
            if backbone_condition(n):
                logging.info(f"[GazePolicy] Backbone parameter: {n} with shape {p.shape}")

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
        # logging.info(f"""
        #     [GazePolicy] Initializing AdamW optimizer with the following parameters:
        #     - Learning Rate: {self.cfg.optimizer_lr}
        #     - Betas: {self.cfg.optimizer_betas}
        #     - Epsilon: {self.cfg.optimizer_eps}
        #     - Weight Decay: {self.cfg.optimizer_weight_decay}
        # """)
        # return torch.optim.AdamW(
        #     params=self.parameters(),
        #     lr=self.cfg.optimizer_lr,
        #     betas=self.cfg.optimizer_betas,
        #     eps=self.cfg.optimizer_eps,
        #     weight_decay=self.cfg.optimizer_weight_decay
        # )

    
    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR | None:
        logging.info(f"""
            [GazePolicy] Initializing scheduler '{self.cfg.scheduler_name}' with the following parameters:
            - Warmup Steps: {self.cfg.scheduler_warmup_steps}
            - Total Training Steps: {num_training_steps}
        """)
        return get_scheduler(
            name=self.cfg.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def get_ema(self):
        logging.info(f"""
            [GazePolicy] EMA is not implemented for GazePolicy.
        """)
        return None
    
    def get_delta_timestamps(self):
        observation_indices = list(reversed(range(0, -self.cfg.n_obs_steps*self.cfg.obs_step_size, -self.cfg.obs_step_size)))
        assert len(observation_indices) == self.cfg.n_obs_steps, "Observation indices length mismatch"
        action_indices = [i - 1 for i in observation_indices] + list(range(self.cfg.horizon))
        gaze_indices = observation_indices + list(range(1, self.cfg.horizon+1))
        logging.info(f"""
            [GazePolicy] Observation Indices: {observation_indices}
            [GazePolicy] Action Indices: {action_indices}
            [GazePolicy] Gaze Indices: {gaze_indices}
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
            # gaze
            **{
                k: [i / self.task_cfg.fps for i in gaze_indices]
                for k in self.cfg.image_to_gaze_key.values()
            },
        }
    
    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        obs_queue_size = (self.cfg.n_obs_steps-1)*self.cfg.obs_step_size+1
        self._queues = {
            self.task_cfg.action_key: deque(maxlen=self.cfg.n_action_steps),
            "action_history": deque(maxlen=obs_queue_size),
            self.task_cfg.state_key: deque(maxlen=obs_queue_size),
            **{key: deque(maxlen=obs_queue_size) for key in self.task_cfg.image_keys},
        }
        self.action_temporal_ensembler.reset()        
        self.gaze_temporal_ensembler.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], return_viz: bool = False) -> Tensor:
        self.eval()
        assert self.task_cfg.state_key in batch, f"Missing {self.task_cfg.state_key} in batch"
        assert all(key in batch for key in self.task_cfg.image_keys), f"Missing {self.task_cfg.image_keys} in batch"
        self._queues = populate_queues(self._queues, batch)

        # if not full action history fill with padding
        obs_max_len = self._queues["action_history"].maxlen
        if len(self._queues["action_history"]) < obs_max_len:
            batch_size = batch[self.task_cfg.state_key].shape[0]
            self._queues["action_history"].extend(
                [self.action_history_padding.unsqueeze(0).expand(batch_size, -1)] * obs_max_len
            )

        viz = {}
        if len(self._queues[self.task_cfg.action_key]) == 0:
            batch = {
                k: torch.stack(list(self._queues[k])[::self.cfg.obs_step_size], dim=1) 
                for k in self.task_cfg.image_keys + [self.task_cfg.state_key, "action_history"]
            }
            batch = self.normalize_inputs(batch)
            actions, viz = self.flow.generate_actions(batch)
            actions = self.unnormalize_outputs({self.task_cfg.action_key: actions})[self.task_cfg.action_key]
            weighted_actions = self.action_temporal_ensembler.update(
                actions[:, :, :self.task_cfg.action_dim],
                n=self.cfg.n_action_steps
            )
            if len(self.cfg.image_to_gaze_key) > 0:
                weighted_actions = torch.cat(
                    [
                        weighted_actions,
                        self.gaze_temporal_ensembler.update(
                            actions[:,:, self.task_cfg.action_dim:],
                            n=self.cfg.n_action_steps
                        )
                    ], 
                    dim=-1
                )
            self._queues[self.task_cfg.action_key].extend(weighted_actions.transpose(0, 1))

        action = self._queues[self.task_cfg.action_key].popleft()
        self._queues["action_history"].append(action)
        
        action = action[:, :self.task_cfg.action_dim] # remove gaze from action

        if return_viz: return action, viz
        return action

    @torch.no_grad()
    def get_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)
        actions, viz = self.flow.generate_actions(batch)
        actions = self.unnormalize_outputs({self.task_cfg.action_key: actions})[self.task_cfg.action_key]
        return actions, viz

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        # add gazes to action history
        batch[self.task_cfg.action_key] = torch.cat(
            [
                batch[self.task_cfg.action_key],
                *[batch[key] for key in self.cfg.image_to_gaze_key.values()]
            ],
            dim=-1
        )
        assert batch[self.task_cfg.action_key].shape[-1] == self.action_dim

        # extract action history
        batch["action_history"] = batch[self.task_cfg.action_key][:, :self.cfg.n_obs_steps]
        batch[self.task_cfg.action_key] = batch[self.task_cfg.action_key][:, self.cfg.n_obs_steps:]

        # convert padded history to zeros
        outside_episode_bound = batch[self.task_cfg.state_key + "_is_pad"]
        batch["action_history"][outside_episode_bound] = self.action_history_padding

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
    def __init__(self, policy_cfg: GazePolicyConfig, task_cfg: TaskConfig):
        super().__init__()
        self.cfg = policy_cfg
        self.task_cfg = task_cfg
        self.action_dim = task_cfg.action_dim + len(policy_cfg.image_to_gaze_key)*2

        self.flow_matcher = _make_flow_matcher(
            policy_cfg.flow_matcher,
            **policy_cfg.flow_matcher_kwargs,
        )

        # observation processing
        
        self.resize = Resize(policy_cfg.resize_shape)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        # for param in self.dino.parameters():
        #     param.requires_grad = False
        self.dino_embed = nn.Linear(self.dino.embed_dim, policy_cfg.dim_model)
        self.gaze_embed = nn.Linear(2, policy_cfg.dim_model)
        self.state_embed = nn.Linear(task_cfg.state_dim, policy_cfg.dim_model)
        self.state_dropout = nn.Dropout(policy_cfg.proprio_dropout)
        self.action_history_embed = nn.Linear(self.action_dim, policy_cfg.dim_model)
        self.action_history_dropout = nn.Dropout(policy_cfg.proprio_dropout)

        # self.transformer_encoder = TransformerEncoder(
        #     hidden_size=policy_cfg.dim_model,
        #     num_layers=policy_cfg.self_attn_n_layers,
        #     num_heads=policy_cfg.n_heads,
        #     mlp_ratio=policy_cfg.mlp_ratio,
        #     dropout=policy_cfg.dropout,
        # )

        self.attn_pooling = AttentionPooling(
            hidden_size=policy_cfg.dim_model,
            out_dim=self.cfg.attn_pooling_out_dim,
            num_queries=self.cfg.attn_pooling_n_queries,
            depth=self.cfg.attn_pooling_n_layers,
            num_heads= self.cfg.n_heads,
            mlp_ratio= policy_cfg.mlp_ratio,
            dropout= policy_cfg.dropout,
        ) 

        self.DiT = DiT(
            in_dim=self.action_dim,
            out_dim=self.action_dim,
            cond_dim=policy_cfg.attn_pooling_out_dim,
            # cond_dim=self.cfg.attn_pooling_out_dim + policy_cfg.n_obs_steps * (task_cfg.state_dim + len(policy_cfg.image_to_gaze_key) * 2),
            hidden_size=policy_cfg.dim_model,
            depth=policy_cfg.n_decoder_layers,
            num_heads=policy_cfg.n_heads,
            mlp_ratio=policy_cfg.mlp_ratio,
            dropout=policy_cfg.dropout,
            time_dim=policy_cfg.time_dim,
        )
        self.register_buffer(
            "dit_pos_embed", 
            get_1d_rotary_embed(policy_cfg.dim_model // policy_cfg.n_heads, torch.arange(policy_cfg.horizon))
        )

    def prepare_global_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        viz = {}
        cond = []
        batch = dict(batch)  # make a copy to avoid modifying the original batch

        # hacky extract gaze data from action history
        gaze_data = torch.split(
            batch["action_history"],
            [self.task_cfg.action_dim] + [2] * len(self.cfg.image_to_gaze_key),
            dim=-1
        )[1:]
        for gaze_key, gaze in zip(self.cfg.image_to_gaze_key.values(), gaze_data):
            if self.training:
                batch[gaze_key] = gaze + torch.randn_like(gaze) * self.cfg.gaze_noise_factor  # add noise
            else:
                batch[gaze_key] = gaze

        # process images
        tokens = []
        pos_embed = []
        for image_key in self.task_cfg.image_keys:
            # resize images
            images = batch[image_key]  # (B, S, C, H, W)
            batch_size = images.shape[0]
            images = einops.rearrange(images, 'b s c h w -> (b s) c h w')
            images = self.resize(images)  # (B, S, C, H, W)
            images = einops.rearrange(images, '(b s) c h w -> b s c h w', b=batch_size)

            # peripheral processing
            peripheral = einops.rearrange(images, 'b s c h w -> (b s) c h w')
            peripheral_feat = self.dino.forward_features(peripheral)["x_norm_patchtokens"]  # (B*S, D)
            peripheral_feat = einops.rearrange(peripheral_feat, '(b s) l d -> b (s l) d', b=batch_size)
            tokens.append(self.dino_embed(peripheral_feat))  # (B, S, D)
            pos_embed.append(
                get_foveated_pos_embed(
                    gaze=torch.zeros((batch_size, self.cfg.n_obs_steps, 2), device=peripheral_feat.device),  # centered for peripheral
                    image_shape=self.cfg.resize_shape,
                    crop_shape=self.cfg.resize_shape,
                    feat_shape=(int(self.cfg.resize_shape[0] / self.dino.patch_size), int(self.cfg.resize_shape[1] / self.dino.patch_size)),
                    n_obs_steps=self.cfg.n_obs_steps,
                    dim=self.cfg.dim_model // self.cfg.n_heads,
                )
            )
            if not self.training:
                viz[image_key] = peripheral

            # crop processing
            if image_key in self.cfg.image_to_gaze_key:
                gaze_key = self.cfg.image_to_gaze_key[image_key]
                foveal = crop_at_center(
                    images=einops.rearrange(images, 'b s c h w -> (b s) c h w'), 
                    centers=einops.rearrange(batch[gaze_key], "b s c -> (b s) c"), 
                    crop_shape=self.cfg.foveal_shape, 
                    out_shape=self.cfg.resize_shape,
                )
                foveal_feat = self.dino.forward_features(foveal)["x_norm_patchtokens"]  # (B*S, D)
                foveal_feat = einops.rearrange(foveal_feat, '(b s) l d -> b (s l) d', b=batch_size)
                tokens.append(self.dino_embed(foveal_feat))  # (B, S, D)
                pos_embed.append(
                    get_foveated_pos_embed(
                        gaze=batch[gaze_key], 
                        image_shape=self.cfg.resize_shape,
                        crop_shape=self.cfg.foveal_shape,
                        feat_shape=(int(self.cfg.resize_shape[0] / self.dino.patch_size), int(self.cfg.resize_shape[1] / self.dino.patch_size)),
                        n_obs_steps=self.cfg.n_obs_steps,
                        dim=self.cfg.dim_model // self.cfg.n_heads,
                    )
                )
                tokens.append(
                    self.gaze_embed(batch[gaze_key])  # (B, S, D)
                )
                pos_embed.append(
                    get_foveated_pos_embed(
                        gaze=batch[gaze_key], 
                        image_shape=self.cfg.resize_shape,
                        crop_shape=self.cfg.foveal_shape,
                        feat_shape=(1,1),
                        n_obs_steps=self.cfg.n_obs_steps,
                        dim=self.cfg.dim_model // self.cfg.n_heads,
                    )
                )
                if not self.training:
                    viz[gaze_key] = foveal

        tokens.append(
            self.state_embed(self.state_dropout(
                batch[self.task_cfg.state_key] 
            ))
        )
        pos_embed.append(
            get_foveated_pos_embed(
                gaze=torch.zeros((batch_size, self.cfg.n_obs_steps, 2), device=batch[self.task_cfg.state_key].device),  # centered for state
                image_shape=self.cfg.resize_shape,
                crop_shape=self.cfg.resize_shape,
                feat_shape=(1,1),
                n_obs_steps=self.cfg.n_obs_steps,
                dim=self.cfg.dim_model // self.cfg.n_heads,
            )
        )
        if self.cfg.use_action_history:
            tokens.append(
                self.action_history_embed(self.action_history_dropout(
                    batch["action_history"]
                ))  # (B, S, D)
            )
            pos_embed.append(
                get_foveated_pos_embed(
                    gaze=torch.zeros((batch_size, self.cfg.n_obs_steps, 2), device=batch["action_history"].device),  # centered for action history
                    image_shape=self.cfg.resize_shape,
                    crop_shape=self.cfg.resize_shape,
                    feat_shape=(1,1),
                    n_obs_steps=self.cfg.n_obs_steps,
                    dim=self.cfg.dim_model // self.cfg.n_heads,
                )
            )



        tokens = torch.cat(tokens, dim=1)  # (B, S, D)
        pos_embed = torch.cat(pos_embed, dim=1).unsqueeze(1)  # (B, 1, S, D//n_heads)
        # tokens = self.transformer_encoder(tokens, pos_embed)  # (B, S, D)
        cond = self.attn_pooling(tokens, pos_embed)


        # cond = torch.cat([
        #     cond,
        #     self.state_dropout(batch[self.task_cfg.state_key].flatten(start_dim=1)),  # (B, S, D)
        #     torch.cat([batch[key].flatten(start_dim=1) for key in self.cfg.image_to_gaze_key.values()], dim=-1)
        # ], dim=-1)  # (B, S, D + state_dim + gaze_dim)

        return cond, viz

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        num_steps = self.cfg.n_sampling_steps
        shape = (batch_size, self.cfg.horizon, self.action_dim)
        x = torch.randn(shape, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        for n in range(num_steps):
            timestep = n * dt * torch.ones(x.shape[0], device=x.device)
            vt = self.DiT(x=x, timestep=timestep, cond=global_cond, pos_embed=self.dit_pos_embed)
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
        vt = self.DiT(x=xt, timestep=timestep, cond=global_cond, pos_embed=self.dit_pos_embed)
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