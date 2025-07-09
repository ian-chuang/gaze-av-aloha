from gaze_av_aloha.policies.gaze_policy.gaze_policy_config import GazePolicyConfig
from gaze_av_aloha.policies.diffusion_policy.diffusion_policy_config import DiffusionPolicyConfig
from gaze_av_aloha.policies.flow_policy.flow_policy_config import FlowPolicyConfig
from gaze_av_aloha.policies.flare_policy.flare_policy_config import FlarePolicyConfig
from gaze_av_aloha.policies.cage_policy.cage_policy_config import CAGEPolicyConfig
from gaze_av_aloha.configs import Config, TaskConfig, TrainConfig, WandBConfig
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="policy", name="base_gaze_policy_config", node=GazePolicyConfig)
cs.store(group="policy", name="base_diffusion_policy_config", node=DiffusionPolicyConfig)
cs.store(group="policy", name="base_flow_policy_config", node=FlowPolicyConfig)
cs.store(group="policy", name="base_flare_policy_config", node=FlarePolicyConfig)
cs.store(group="policy", name="base_cage_policy_config", node=CAGEPolicyConfig)
cs.store(group="task", name="base_task_config", node=TaskConfig) 
cs.store(group="train", name="base_train_config", node=TrainConfig)
cs.store(group="wandb", name="base_wandb_config", node=WandBConfig)