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

roslaunch av_aloha 2arms_teleop_no_left.launch use_sim:=true use_rviz:=true
