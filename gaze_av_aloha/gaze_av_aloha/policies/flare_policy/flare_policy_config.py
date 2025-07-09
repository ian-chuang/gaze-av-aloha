from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class FlarePolicyConfig(PolicyConfig):
    type: str = "flare_policy"

    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "mean_std"
    action_norm_mode: str = "mean_std"

    action_horizon: int = 8
    pred_horizon: int = 16
    obs_horizon: int = 1

    resize_shape: tuple[int, int] = (240, 320)
    crop_shape: tuple[int, int] = (216, 288)

    observer_name: str = "resnet18"
    observer_tokenize: bool = False

    flow_matcher_name: str = "target"
    flow_matcher_sigma: float = 0.0
    flow_matcher_num_sampling_steps: int = 6

    flow_net_block_type: str = "adaln"  # adaln, cross
    flow_net_hidden_dim: int = 512
    flow_net_num_layers: int = 8
    flow_net_num_heads: int = 8
    flow_net_mlp_ratio: int = 4
    flow_net_dropout: float = 0.1

    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500