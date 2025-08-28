#!/bin/bash

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh

# Change directory to the project folder
cd ~/GitHub/gaze-av-aloha

conda activate gym_av
# Train for av_aloha_hang_ring task
# python gaze_av_aloha/scripts/train.py \
#   policy=foveated_vit_policy \
#   task=av_aloha_hang_ring \
#   policy.use_gaze_as_action=false \
#   policy.gaze_model_repo_id=Jinyu220/gaze_model_av_aloha_real_hang_circle_v2 \
#   policy.vision_encoder_kwargs.repo_id=iantc104/mae_vitb_foveated_vit \
#   policy.optimizer_lr_backbone=1e-5 \
#   wandb.enable=true \
#   wandb.project=hang_augementation_hang_ringv4_unet \
#   wandb.entity=jinyuzou220-uc-davis \
#   wandb.job_name=fov-unet-augementation_hangv4_ring \
#   device=cuda

# # Train for av_aloha_put_coin task
# python gaze_av_aloha/scripts/train.py \
#   policy=foveated_vit_policy \
#   task=av_aloha_put_coin \
#   policy.use_gaze_as_action=false \
#   policy.gaze_model_repo_id=Jinyu220/gaze_model_av_aloha_real_put_coin_v2 \
#   policy.vision_encoder_kwargs.repo_id=iantc104/mae_vitb_foveated_vit \
#   policy.optimizer_lr_backbone=1e-5 \
#   wandb.enable=true \
#   wandb.project=hang_augementation_put_coinv4_unet \
#   wandb.entity=jinyuzou220-uc-davis \
#   wandb.job_name=fov-unet-augementation_put_coinv4 \
#   device=cuda



python gaze_av_aloha/scripts/train.py \
  task=av_aloha_put_coin \
  policy=vit_policy \
  policy.vision_encoder_kwargs.repo_id=iantc104/mae_vitb_vit \
  policy.optimizer_lr_backbone=1e-5 \
  wandb.enable=true \
  wandb.project=augementation_put_coinv5-fine \
  wandb.entity=jinyuzou220-uc-davis\
  wandb.job_name=fine-augementation_put_coin8025 \
  device=cuda


python gaze_av_aloha/scripts/train.py \
  task=av_aloha_hang_ring \
  policy=vit_policy \
  policy.vision_encoder_kwargs.repo_id=iantc104/mae_vitb_vit \
  policy.optimizer_lr_backbone=1e-5 \
  wandb.enable=true \
  wandb.project=augementation_hang_ring53-fine \
  wandb.entity=jinyuzou220-uc-davis\
  wandb.job_name=fine-augementation_hang_ring25\
  device=cuda