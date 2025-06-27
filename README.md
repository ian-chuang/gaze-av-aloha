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


python train.py policy=flow_policy task=av_aloha_sim_hook_package policy.use_gaze=true device=cuda:0

python train.py policy=flow_policy task=av_aloha_sim_thread_needle policy.gaze_prob=0.5 device=cuda:0
python train.py policy=flow_policy task=av_aloha_sim_thread_needle policy.gaze_prob=0.0 device=cuda:1