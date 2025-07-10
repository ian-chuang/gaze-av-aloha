from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class CAGEPolicyConfig(PolicyConfig):
    type: str = "cage_policy"

    n_obs_steps: int = 1
    obs_step_size: int = 1
    n_action_steps: int = 2
    horizon: int = 16
    use_temporal_ensemble: bool = True
    temporal_ensemble_coeff: float = -0.01
    drop_n_last_frames: int = 6
    
    # Observation
    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "mean_std" 
    action_norm_mode: str = "mean_std" 
    image_to_gaze_key: dict[str, str] = field(default_factory=lambda: {})

    # vision
    input_shape: tuple = (240, 320)
    vision_encoder: str = "vit"  # options: resnet, dino, vit
    vision_encoder_kwargs: dict = field(default_factory=lambda: {})

    # gaze
    use_gaze: bool = False
    gaze_noise: float = 0.02

    # Attention
    dim_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1

    # Pool
    pool_n_queries: int = 16
    pool_n_layers: int = 4

    # DiT
    dit_time_dim: int = 256
    dit_n_layers: int = 8

    # Flow Matching
    n_sampling_steps: int = 6

    # Training
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.99