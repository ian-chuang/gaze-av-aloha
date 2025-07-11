# gaze-av-aloha

```
git clone https://github.com/ian-chuang/gaze-av-aloha.git
git submodule init
git submodule update
conda create -n gaze python=3.10
pip install -e ./gym_av_aloha
pip install git+https://github.com/huggingface/lerobot.git@483be9aac217c2d8ef16982490f22b2ad091ab46
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

# EVAL

### foveated vit
MUJOCO_EGL_DEVICE_ID=0 python train.py policy=foveated_vit_policy task=av_aloha_sim_thread_needle wandb.job_name=thread_foveated device=cuda:0
MUJOCO_EGL_DEVICE_ID=1 python train.py policy=foveated_vit_policy task=av_aloha_sim_pour_test_tube wandb.job_name=pour_foveated device=cuda:1
MUJOCO_EGL_DEVICE_ID=2 python train.py policy=foveated_vit_policy task=av_aloha_sim_hook_package wandb.job_name=hook_foveated device=cuda:2
MUJOCO_EGL_DEVICE_ID=3 python train.py policy=foveated_vit_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_foveated device=cuda:3
MUJOCO_EGL_DEVICE_ID=0 python train.py policy=foveated_vit_policy task=av_aloha_sim_cube_transfer wandb.job_name=cube_foveated device=cuda:0
MUJOCO_EGL_DEVICE_ID=1 python train.py policy=foveated_vit_policy task=av_aloha_sim_peg_insertion wandb.job_name=peg_foveated device=cuda:1

### vit
MUJOCO_EGL_DEVICE_ID=0 python train.py policy=vit_policy task=av_aloha_sim_thread_needle wandb.job_name=thread_vit device=cuda:0
MUJOCO_EGL_DEVICE_ID=1 python train.py policy=vit_policy task=av_aloha_sim_pour_test_tube wandb.job_name=pour_vit device=cuda:1
MUJOCO_EGL_DEVICE_ID=2 python train.py policy=vit_policy task=av_aloha_sim_hook_package wandb.job_name=hook_vit device=cuda:2
MUJOCO_EGL_DEVICE_ID=3 python train.py policy=vit_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_vit device=cuda:3
MUJOCO_EGL_DEVICE_ID=0 python train.py policy=vit_policy task=av_aloha_sim_cube_transfer wandb.job_name=cube_vit device=cuda:0
MUJOCO_EGL_DEVICE_ID=1 python train.py policy=vit_policy task=av_aloha_sim_peg_insertion wandb.job_name=peg_vit device=cuda:1

### low res vit

MUJOCO_EGL_DEVICE_ID=0 python train.py policy=low_res_vit_policy task=av_aloha_sim_thread_needle wandb.job_name=thread_low_res_vit device=cuda:0
MUJOCO_EGL_DEVICE_ID=1 python train.py policy=low_res_vit_policy task=av_aloha_sim_pour_test_tube wandb.job_name=pour_low_res_vit device=cuda:1
MUJOCO_EGL_DEVICE_ID=2 python train.py policy=low_res_vit_policy task=av_aloha_sim_hook_package wandb.job_name=hook_low_res_vit device=cuda:2
MUJOCO_EGL_DEVICE_ID=3 python train.py policy=low_res_vit_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_low_res_vit device=cuda:3
MUJOCO_EGL_DEVICE_ID=0 python train.py policy=low_res_vit_policy task=av_aloha_sim_cube_transfer wandb.job_name=cube_low_res_vit device=cuda:0
MUJOCO_EGL_DEVICE_ID=1 python train.py policy=low_res_vit_policy task=av_aloha_sim_peg_insertion wandb.job_name=peg_low_res_vit device=cuda:1