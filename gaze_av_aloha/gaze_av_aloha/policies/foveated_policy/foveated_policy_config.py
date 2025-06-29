from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class FoveatedPolicyConfig(PolicyConfig):
    type: str = "foveated_policy"

    n_obs_steps: int = 1
    obs_step_size: int = 1
    n_action_steps: int = 2
    horizon: int = 16
    use_temporal_ensemble: bool = True
    temporal_ensemble_coeff: float = 0.0
    drop_n_last_frames: int = 6
    
    # Observation
    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "min_max" 
    action_norm_mode: str = "min_max" 
    image_to_gaze_key: dict[str, str] = field(default_factory=lambda: {})
    input_shape: tuple = (480, 640) 
    crop_shape: tuple = (432, 576)
    resize_shape: tuple = (960, 1280)
    
    backbone: str = "stt"
    freeze_n_layers: int = 6
    gaze_noise: float = 0.02
    use_gaze: bool = True
    use_action_history: bool = False

    # Transformer Layers
    dim_model: int = 512
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Attention Pooling
    pool_n_queries: int = 16
    pool_n_layers: int = 2

    # DiT
    dit_n_layers: int = 8
    dit_time_dim: int = 128

    # Flow Matching
    n_sampling_steps: int = 10

    # Training
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.99