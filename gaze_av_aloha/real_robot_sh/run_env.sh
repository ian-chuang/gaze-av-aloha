#!/bin/bash
# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh
# Change directory to the project folder
cd ~/GitHub/gaze-av-aloha

# Deactivate any active Conda environments
conda deactivate
conda deactivate
conda deactivate

# Source ROS Noetic setup
source /opt/ros/noetic/setup.sh

# Source the Interbotix workspace setup
source interbotix_ws/devel/setup.sh

# Activate the Conda environment
conda activate gym_av

# Change directory to the robot folder
cd gaze_av_aloha/gaze_av_aloha/robot

# Run the Python script
python env.py
