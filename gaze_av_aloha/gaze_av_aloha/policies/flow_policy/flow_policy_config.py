from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class FlowPolicyConfig(PolicyConfig):
    type: str = "flow_policy"

    n_obs_steps: int = 1
    obs_step_size: int = 1
    n_action_steps: int = 2
    horizon: int = 16
    temporal_ensemble_coeff: float = 0.0
    drop_n_last_frames: int = 6
    
    # Observation
    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "min_max" 
    action_norm_mode: str = "min_max" 
    resize_shape: tuple = (240, 320)
    input_shape: tuple = (224,224)
    dino_freeze_n_layers: int = 0
    state_dropout: float = 0.1 # need to confirm this is valid

    # Transformer Layers
    dim_model: int = 512
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Attention Pooling
    pool_n_queries: int = 4
    pool_out_dim: int = 512
    pool_n_layers: int = 4

    # DiT
    dit_n_layers: int = 8
    time_dim: int = 128

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