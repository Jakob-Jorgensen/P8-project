#!/bin/bash

# List of ROS 2 commands to run in separate terminals
commands=(
  "ros2 run gg_cnn gg_cnn_image_processing"
  "ros2 run communication_package udp_reciver" 
  "ros2 run communication_package udp_sender"
  "ros2 run vlm vlm_interface"
  "ros2 run llm llm_interface"
  "ros2 launch realsense2_camera rs_launch.py config_file:=/home/max/Documents/came_settings.yaml" 
 "ros2 run speech2text speech_node" 
 "ros2 run speech2text ux_node"
)

# Launch each command in a new terminal
for cmd in "${commands[@]}"; do
  gnome-terminal -- bash -c "$cmd; exec bash"
done

