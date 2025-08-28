#!/bin/bash

# 激活你的 conda 环境（可选，根据你的实际环境修改）
# conda activate gym_av
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gym_av
cd /home/jinyu/GitHub/gaze-av-aloha/gaze_av_aloha/scripts
# 依次运行训练脚本
python train_gaze_model_argparse.py --task insert_square_v2 --dataset Jinyu220/put_square_21
python train_gaze_model_argparse.py --task put_coin_v2 --dataset Jinyu220/coin_2
python train_gaze_model_argparse.py --task hang_circle_v2 --dataset Jinyu220/circle_2
python train_gaze_model_argparse.py --task put_tube_v2 --dataset Jinyu220/put_tube_singlev2

echo "All tasks finished!"