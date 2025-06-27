from dataclasses import field, dataclass
from gaze_av_aloha.configs import PolicyConfig

@dataclass
class GazePolicyConfig(PolicyConfig):
    type: str = "gaze_policy"

    n_obs_steps: int = 1
    obs_step_size: int = 1
    n_action_steps: int = 2 # need to confirm this is valid
    horizon: int = 16
    action_temporal_ensemble_coeff: float = 0.0 # need to confirm this is valid
    gaze_temporal_ensemble_coeff: float = 0.0 # need to confirm this is valid
    drop_n_last_frames: int = 6
    
    # Observation
    image_norm_mode: str = "mean_std"
    state_norm_mode: str = "min_max" # need to confirm this is valid
    action_norm_mode: str = "min_max" # need to confirm this is valid
    resize_shape: tuple = (240,320)
    input_shape: tuple = (224,224)
    use_action_history: bool = False # need to confirm this is valid
    action_history_padding: list[float] = field(default_factory=lambda: [
        0, -0.082, 1.06, 0, -0.953, 0, 1, 0, -0.082, 1.06, 0, -0.953, 0, 1, 0, -0.6, 0.5, 0, 0.5, 0, 0
    ]) 
    proprio_dropout: float = 0.1 # need to confirm this is valid

    # gaze
    image_to_gaze_key: dict[str, str] = field(default_factory=lambda: {}) # need to confirm this is valid
    foveal_shape: tuple = (84, 112)
    gaze_noise_factor: float = 0.02 # please confirm this is valid

    # Image Pooling
    # self_attn_n_layers: int = 4
    attn_pooling_n_queries: int = 16
    attn_pooling_n_layers: int = 4
    
    # Transformer
    n_decoder_layers: int = 8
    dim_model: int = 432 
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    time_dim: int = 128

    # Flow Matching
    flow_matcher: str = "target"
    n_sampling_steps: int = 10
    flow_matcher_kwargs: dict = field(default_factory=lambda: {
        "sigma": 0.0,
    })

    # Training
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    use_ema: bool = False
    ema_decay: float = 0.99


