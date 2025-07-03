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


python train.py policy=no_gaze_policy task=av_aloha_sim device=cuda:0 wandb.job_name=thread_1obs policy.n_obs_steps=1
python train.py policy=gaze_policy task=av_aloha_sim device=cuda:0 wandb.job_name=thread_gaze_1obs policy.n_obs_steps=1
python train.py policy=gaze_policy task=av_aloha_sim device=cuda:1 wandb.job_name=thread_gaze_2-2obs policy.n_obs_steps=2 policy.obs_step_size=2
python train.py policy=gaze_policy task=av_aloha_sim device=cuda:1 wandb.job_name=thread_gaze_4-2obs policy.n_obs_steps=4 policy.obs_step_size=2

