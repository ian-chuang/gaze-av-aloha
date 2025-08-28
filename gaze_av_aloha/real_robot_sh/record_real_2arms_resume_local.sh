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

# Change directory to the robot_scripts folder
cd /home/jinyu/GitHub/gaze-av-aloha/gaze_av_aloha/robot_scripts

# Activate the Conda environment
conda activate gym_av

# Run the Python script
python record_real_episodes_no_left_resume_local.py
