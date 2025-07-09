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
from gaze_av_aloha.policies.flare_policy.flare_policy_config import FlarePolicyConfig
from gaze_av_aloha.policies.flare_policy.flow import FlowTransformer
from gaze_av_aloha.policies.flare_policy.flow_matchers import get_flow_matcher
from gaze_av_aloha.policies.flare_policy.observer import ResNetObserver
from gaze_av_aloha.policies.normalize import Normalize, Unnormalize

class FlarePolicy(Policy):
    def __init__(
        self, 
        policy_cfg: FlarePolicyConfig,
        task_cfg: TaskConfig,
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()

        self.cfg = policy_cfg
        self.task_cfg = task_cfg
        self.stats = stats
        self.num_sampling_steps = policy_cfg.flow_matcher_num_sampling_steps
        self.pred_horizon = policy_cfg.pred_horizon
        self.action_horizon = policy_cfg.action_horizon
        self.action_dim = task_cfg.action_dim

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

        self._action_queue = None
        self.observer = ResNetObserver(
            state_key=task_cfg.state_key,
            image_keys=task_cfg.image_keys,
            resize_shape=policy_cfg.resize_shape,
            crop_shape=policy_cfg.crop_shape,
            state_dim=task_cfg.state_dim,
            tokenize=policy_cfg.observer_tokenize,
        )
        self.obs_dim = len(task_cfg.image_keys) * 512 + task_cfg.state_dim
        self.FM = get_flow_matcher(
            name=policy_cfg.flow_matcher_name,
            sigma=policy_cfg.flow_matcher_sigma,
            num_sampling_steps=self.num_sampling_steps,
        )
        self.flow_net = self._init_flow_net(condition_dim=self.obs_dim)

        self.reset()

    def _init_flow_net(self, condition_dim):
        """
        Initialize the velocity prediction network
        """
        return FlowTransformer(
            input_dim=self.task_cfg.action_dim,
            condition_dim=condition_dim,
            output_dim=self.task_cfg.action_dim,
            hidden_dim=self.cfg.flow_net_hidden_dim,
            num_layers=self.cfg.flow_net_num_layers,
            num_heads=self.cfg.flow_net_num_heads,
            block_type=self.cfg.flow_net_block_type,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.observer(batch)
        target = batch[self.task_cfg.action_key]
        loss, metrics = self.FM.compute_loss(self.flow_net, target=target, cond=features)
        metrics['flow_loss'] = loss.item()
        return loss, metrics

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch[self.task_cfg.state_key].shape[0]
        features = self.observer(batch)
        actions = self.FM.sample(
            self.flow_net,
            (batch_size, self.pred_horizon, self.action_dim),
            features.device,
            self.num_sampling_steps,
            cond=features,
            return_traces=False
        )

        return actions

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor], return_viz: bool = False) -> torch.Tensor:
        self.eval()
        batch = {k: v.unsqueeze(1) for k, v in batch.items() if k in self.task_cfg.image_keys + [self.task_cfg.state_key]}
        batch = self.normalize_inputs(batch)
        if len(self._action_queue) == 0:
            actions = self.generate_actions(batch)
            actions = actions[:, :self.action_horizon]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        if return_viz: 
            return self._action_queue.popleft(), {}
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, metrics = self.compute_loss(batch)
        metrics['flow_loss'] = loss.item()
        return loss, metrics

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

    # def get_action_indices(self):
    #     return list(range(self.pred_horizon))

    # def get_observation_indices(self):
    #     return [0]
    
    def get_delta_timestamps(self):
        action_indices = list(range(self.pred_horizon))
        observation_indices = [0]
        return {
            **{
                k: [i / self.task_cfg.fps for i in observation_indices]
                for k in self.task_cfg.image_keys
            },
            self.task_cfg.state_key: [i / self.task_cfg.fps for i in observation_indices],
            self.task_cfg.action_key: [i / self.task_cfg.fps for i in action_indices],
        }

    def reset(self):
        self._action_queue = deque([], maxlen=self.action_horizon)

