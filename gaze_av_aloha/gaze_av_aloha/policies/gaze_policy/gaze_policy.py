import torch
from torch import Tensor, nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from huggingface_hub import PyTorchModelHubMixin

from gaze_av_aloha.configs import TaskConfig
from gaze_av_aloha.policies.policy import Policy
from gaze_av_aloha.policies.gaze_policy.gaze_policy_config import GazePolicyConfig
from gaze_av_aloha.policies.gaze_policy.flow import FlowTransformer
from gaze_av_aloha.policies.gaze_policy.flow_matchers import get_flow_matcher
from gaze_av_aloha.policies.gaze_policy.observer import ResNetObserver, ViTObserver
from gaze_av_aloha.policies.normalize import Normalize, Unnormalize

from gaze_av_aloha.utils.policy_utils import (
    populate_queues,
)

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
        self.num_sampling_steps = policy_cfg.flow_matcher_num_sampling_steps
        self.pred_horizon = policy_cfg.horizon
        self.action_dim = task_cfg.action_dim

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

        self.target_keys_to_dim = {
            task_cfg.action_key: task_cfg.action_dim,
            **{k: 2 for k in policy_cfg.image_to_gaze_key.values() if policy_cfg.use_gaze},
        }
        self.noise_dim = sum(self.target_keys_to_dim.values())

        self._action_queue = None
        if policy_cfg.observer_name == "resnet":
            self.observer = ResNetObserver(
                state_key=task_cfg.state_key,
                image_keys=list(policy_cfg.image_to_gaze_key.keys()),
                resize_shape=policy_cfg.resize_shape,
                crop_shape=policy_cfg.crop_shape,
                state_dim=task_cfg.state_dim,
                tokenize=policy_cfg.observer_tokenize,
            )
        elif policy_cfg.observer_name == "vit":
            self.observer = ViTObserver(policy_cfg, task_cfg)
        else:
            raise ValueError(f"Unknown observer name: {policy_cfg.observer_name}")
        self.obs_dim = len(policy_cfg.image_to_gaze_key) * 512 + task_cfg.state_dim
        self.FM = get_flow_matcher(
            name=policy_cfg.flow_matcher_name,
            sigma=policy_cfg.flow_matcher_sigma,
            num_sampling_steps=self.num_sampling_steps,
        )
        
        self.flow_net = self._init_flow_net(condition_dim=self.obs_dim)

        if policy_cfg.use_temporal_ensemble:
            self.temporal_ensembler = TemporalEnsembler(
                temporal_ensemble_coeff=policy_cfg.temporal_ensemble_coeff,
                chunk_size=policy_cfg.horizon,
            )

        self.reset()

    def _init_flow_net(self, condition_dim):
        """
        Initialize the velocity prediction network
        """
        return FlowTransformer(
            input_dim=self.noise_dim,
            condition_dim=condition_dim,
            output_dim=self.noise_dim,
            hidden_dim=self.cfg.flow_net_hidden_dim,
            num_layers=self.cfg.flow_net_num_layers,
            num_heads=self.cfg.flow_net_num_heads,
            block_type=self.cfg.flow_net_block_type,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:

        x1 = []
        for key in self.target_keys_to_dim:
            x1.append(batch[key][:, self.cfg.n_obs_steps:])
            batch[key] = batch[key][:, :self.cfg.n_obs_steps]
        x1 = torch.cat(x1, dim=-1)

        features = self.observer(batch)
        target = x1
        loss, metrics = self.FM.compute_loss(self.flow_net, target=target, cond=features)
        metrics['flow_loss'] = loss.item()
        return loss, metrics

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch[self.task_cfg.state_key].shape[0]
        features = self.observer(batch)
        actions = self.FM.sample(
            self.flow_net,
            (batch_size, self.pred_horizon, self.noise_dim),
            features.device,
            self.num_sampling_steps,
            cond=features,
            return_traces=False
        )

        outputs = {
            k: v
            for k, v in zip(
                self.target_keys_to_dim.keys(),
                torch.split(actions, list(self.target_keys_to_dim.values()), dim=-1),
            )
        }

        return outputs, {}

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor], return_viz: bool = False) -> torch.Tensor:
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
            outputs, viz = self.generate_actions(batch)
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

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)

        # pad gaze obs with zeros
        for gaze_key in self.cfg.image_to_gaze_key.values():
            is_pad = batch[self.task_cfg.action_key + "_is_pad"][:, :self.cfg.n_obs_steps] # use action key since it is 1 behind
            batch[gaze_key][:, :self.cfg.n_obs_steps][is_pad] = 0.0
        
        # pad action obs with state TODO not sure this is correct
        is_pad = batch[self.task_cfg.action_key + "_is_pad"][:, :self.cfg.n_obs_steps]
        state_pad = batch[self.task_cfg.state_key][:, :self.cfg.n_obs_steps][is_pad]
        batch[self.task_cfg.action_key][:, :self.cfg.n_obs_steps][is_pad] = state_pad

        loss, loss_dict = self.compute_loss(batch)
        return loss, loss_dict

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.cfg.optimizer_lr,
            betas=self.cfg.optimizer_betas,
            eps=self.cfg.optimizer_eps,
            weight_decay= self.cfg.optimizer_weight_decay,
        )

    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR | None:
        return get_scheduler(
            name=self.cfg.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def get_ema(self):
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
        # action_indices = list(range(self.pred_horizon))
        # observation_indices = [0]
        # return {
        #     **{
        #         k: [i / self.task_cfg.fps for i in observation_indices]
        #         for k in self.task_cfg.image_keys
        #     },
        #     self.task_cfg.state_key: [i / self.task_cfg.fps for i in observation_indices],
        #     self.task_cfg.action_key: [i / self.task_cfg.fps for i in action_indices],
        # }

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