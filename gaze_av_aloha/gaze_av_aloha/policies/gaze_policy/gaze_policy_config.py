from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class GazePolicyConfig(PolicyConfig):
    type: str = "gaze_policy"

    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "mean_std"
    action_norm_mode: str = "mean_std"

    image_to_gaze_key: dict[str, str] = field(default_factory=lambda: {})
    use_temporal_ensemble: bool = True
    temporal_ensemble_coeff: float = -0.01

    n_obs_steps: int = 1
    obs_step_size: int = 1
    n_action_steps: int = 2
    horizon: int = 16

    # vit
    input_shape: tuple[int, int] = (240, 320)
    patch_size: int = 16
    vit_input_shape: tuple = (224, 224) #(288, 288)

    use_gaze: bool = False
    strides: tuple[int] = (1, 2, 6)
    grid_sizes: tuple[int] = (2, 3, 3)
    gaze_noise: float = 0.0

    use_crop: bool = True # only when gaze is not used
    crop_shape: tuple[int] = (216, 288)
    dropout: float = 0.1
    dim_model: int = 512
    num_heads: int = 8
    mlp_ratio: int = 4
    pool_n_queries: int = 16
    pool_n_layers: int = 2


    resize_shape: tuple[int, int] = (240, 320)

    observer_name: str = "resnet" # resnet | vit | vit_gaze
    observer_tokenize: bool = True

    flow_matcher_name: str = "target"
    flow_matcher_sigma: float = 0.0
    flow_matcher_num_sampling_steps: int = 6

    flow_net_block_type: str = "cross_adaln"  # adaln, cross
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