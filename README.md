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

# cage policy 

MUJOCO_EGL_DEVICE_ID=0 python train.py policy=cage_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_cage_foveated_vit device=cuda:0 

MUJOCO_EGL_DEVICE_ID=1 python train.py policy=cage_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_cage_vit device=cuda:1

MUJOCO_EGL_DEVICE_ID=2 python train.py policy=cage_policy task=av_aloha_sim_slot_insertion wandb.job_name=slot_cage_dino device=cuda:2
