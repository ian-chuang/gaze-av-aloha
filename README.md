# gaze-av-aloha

```
git clone https://github.com/ian-chuang/gaze-av-aloha.git
git submodule init
git submodule update
conda create -n gaze python=3.10
pip install -e ./gym_av_aloha
pip install git+https://github.com/huggingface/lerobot.git
pip install -e ./gaze_av_aloha
```

# Create Dataset

```
cd gym_av_aloha/scripts
# modify the convert_lerobot_to_avaloha.py file to whatever dataset you want to create
python convert_lerobot_to_avaloha.py
```

```
cd gaze_av_aloha/scripts
python train.py policy=gaze_policy task=av_aloha_sim_hook_package wandb.enable=false 
```

wandb.enable=false train.eval_freq=100 train.viz_freq=99


# gaze

MUJOCO_EGL_DEVICE_ID=0 python train.py policy=gaze_policy task=av_aloha_sim_peg_insertion wandb.job_name=peg_gaze_1obs device=cuda:0 

MUJOCO_EGL_DEVICE_ID=1 python train.py policy=gaze_policy task=av_aloha_sim_pour_test_tube wandb.job_name=pour_gaze_1obs device=cuda:1 

MUJOCO_EGL_DEVICE_ID=2 python train.py policy=gaze_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_gaze_1obs device=cuda:2 

MUJOCO_EGL_DEVICE_ID=3 python train.py policy=gaze_policy task=av_aloha_sim_hook_package wandb.job_name=hook_gaze_1obs device=cuda:3

# flow policy unet

MUJOCO_EGL_DEVICE_ID=0 python train.py policy=flow_policy task=av_aloha_sim_peg_insertion wandb.job_name=peg_flow_1obs device=cuda:0 

MUJOCO_EGL_DEVICE_ID=1 python train.py policy=flow_policy task=av_aloha_sim_pour_test_tube wandb.job_name=pour_flow_1obs device=cuda:1 

MUJOCO_EGL_DEVICE_ID=2 python train.py policy=flow_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_flow_1obs device=cuda:2 

MUJOCO_EGL_DEVICE_ID=3 python train.py policy=flow_policy task=av_aloha_sim_hook_package wandb.job_name=hook_flow_1obs device=cuda:3

# flare policy 

MUJOCO_EGL_DEVICE_ID=0 python train.py policy=flare_policy task=av_aloha_sim_peg_insertion policy.flow_matcher_name=conditional wandb.job_name=peg_flare_conditional device=cuda:0 

MUJOCO_EGL_DEVICE_ID=1 python train.py policy=flare_policy task=av_aloha_sim_pour_test_tube policy.flow_matcher_name=conditional wandb.job_name=pour_flare_conditional device=cuda:1 

MUJOCO_EGL_DEVICE_ID=2 python train.py policy=gaze_policy task=av_aloha_sim_slot_insertion policy.flow_matcher_name=conditional wandb.job_name=slot_gaze_conditional_dino device=cuda:2 

MUJOCO_EGL_DEVICE_ID=3 python train.py policy=gaze_policy task=av_aloha_sim_hook_package policy.flow_matcher_name=conditional wandb.job_name=hook_gaze_conditional device=cuda:3 policy.use_gaze=true train.eval_freq=100


MUJOCO_EGL_DEVICE_ID=2 python train.py policy=gaze_policy task=av_aloha_sim_slot_insertion policy.flow_matcher_name=conditional wandb.job_name=slot_gaze_conditional_resnet_notempens device=cuda:2 

MUJOCO_EGL_DEVICE_ID=3 python train.py policy=gaze_policy task=av_aloha_sim_slot_insertion policy.flow_matcher_name=conditional wandb.job_name=slot_gaze_conditional_resnet_tempens device=cuda:3