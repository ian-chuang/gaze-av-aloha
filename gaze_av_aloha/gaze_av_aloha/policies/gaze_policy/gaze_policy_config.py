from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class GazePolicyConfig(PolicyConfig):
    type: str = "gaze_policy"

    n_obs_steps: int = 1
    obs_step_size: int = 1
    n_action_steps: int = 2
    horizon: int = 16
    use_temporal_ensemble: bool = True
    temporal_ensemble_coeff: float = 0.0 # MAYBE?
    drop_n_last_frames: int = 6
    
    # Observation
    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "mean_std" 
    action_norm_mode: str = "mean_std" 
    image_to_gaze_key: dict[str, str] = field(default_factory=lambda: {})
    input_shape: tuple = (240, 320)

    # backbone
    vision_encoder: str = "vit"  # options: resnet, dino, vit
    vision_encoder_kwargs: dict = field(default_factory=lambda: {})

    # gaze
    use_gaze: bool = True
    gaze_noise: float = 0.0

    # Transformer Layers
    dim_model: int = 512
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Attention Pooling
    pool_n_queries: int = 16
    pool_n_layers: int = 4

    # DiT
    dit_n_layers: int = 8
    dit_time_dim: int = 128

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