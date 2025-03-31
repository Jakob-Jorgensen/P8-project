"""
Program to validate neural network grasp predictions

Workflow:
(1) Load objects and render depth images
(2) Input to network, obtain grasp predictions
(3) Move camera above grasp point, calculate maximum grasp depth
(4) Execute grasp
"""

import pybullet as p
import time
import cv2
import numpy as np
import sys
import scipy.io as scio
from utils.simEnv import SimEnv
import utils.tool as tool
import utils.panda_sim_grasp_arm as PandaSim
from utils.camera import Camera
from ggcnn.ggcnn import GGCNNNet, drawGrasps, drawRect, getGraspDepth

# Constants for gripper finger dimensions
FINGER_L1 = 0.015  # Length of finger segment 1
FINGER_L2 = 0.005  # Length of finger segment 2

def run(database_path, start_idx, objs_num):
    # Connect to PyBullet physics server
    cid = p.connect(p.GUI)  
    # Initialize robot arm at position [0, -0.6, 0]
    panda = PandaSim.PandaSimAuto(p, [0, -0.6, 0])  
    # Initialize simulation environment with robot ID
    env = SimEnv(p, database_path, panda.pandaId) 
    # Initialize camera
    camera = Camera()   
    # Load GGCNN neural network model
    ggcnn = GGCNNNet('ggcnn/ckpt/epoch_0213_acc_0.6374.pth', device="cpu")    

    # Track grasp success statistics
    success_grasp = 0  # Successful grasps counter
    sum_grasp = 0      # Total grasp attempts counter
    tt = 5             # Simulation steps between renders
    
    # Load objects from URDF files
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0  # Counter for consecutive failed grasps
    
    while True:
        # Let physics stabilize
        for _ in range(240*5):  # Run simulation for 5 seconds
            p.stepSimulation()
            
        # Get depth image from camera
        camera_depth = env.renderCameraDepthImage()
        camera_depth = env.add_noise(camera_depth)  # Add realistic noise

        # Predict grasp parameters using GGCNN
        row, col, grasp_angle, grasp_width_pixels = ggcnn.predict(camera_depth, input_size=300)
        grasp_width = camera.pixels_TO_length(grasp_width_pixels, camera_depth[row, col])

        # Convert image coordinates to world coordinates
        grasp_x, grasp_y, grasp_z = camera.img2world([col, row], camera_depth[row, col])
        # Convert finger lengths to pixels
        finger_l1_pixels = camera.length_TO_pixels(FINGER_L1, camera_depth[row, col])
        finger_l2_pixels = camera.length_TO_pixels(FINGER_L2, camera_depth[row, col])
        # Calculate grasp depth
        grasp_depth = getGraspDepth(camera_depth, row, col, grasp_angle, grasp_width_pixels, 
                                  finger_l1_pixels, finger_l2_pixels)
        grasp_z = max(0.7 - grasp_depth, 0)
        
        # Print grasp parameters
        print('*' * 100)
        print('grasp pose:')
        print('grasp_x = ', grasp_x)
        print('grasp_y = ', grasp_y)
        print('grasp_z = ', grasp_z)
        print('grasp_depth = ', grasp_depth)
        print('grasp_angle = ', grasp_angle)
        print('grasp_width = ', grasp_width)
        print('*' * 100)

        # Visualize grasp configuration
        im_rgb = tool.depth2Gray3(camera_depth)
        im_grasp = drawGrasps(im_rgb, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')
        cv2.imshow('im_grasp', im_grasp)
        cv2.waitKey(30)

        # Execute grasp
        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t % tt == 0:
                time.sleep(1./240.)
            
            if panda.step([grasp_x, grasp_y, grasp_z], grasp_angle, grasp_width/2):
                t = 0
                break

        # Evaluate grasp success
        sum_grasp += 1
        if env.evalGraspAndRemove(z_thresh=0.2):  # Check if object lifted above threshold
            success_grasp += 1
            continue_fail = 0
            if env.num_urdf == 0:  # If all objects cleared
                p.disconnect()
                return success_grasp, sum_grasp
        else:
            continue_fail += 1
            if continue_fail == 5:  # Stop after 5 consecutive failures
                p.disconnect()
                return success_grasp, sum_grasp
        
        # Reset arm position
        panda.setArmPos([0.5, -0.6, 0.2])

if __name__ == "__main__":
    start_idx = 0       # Starting index for loading objects
    objs_num = 15       # Number of objects in scene
    database_path = 'models/'
    success_grasp, all_grasp = run(database_path, start_idx, objs_num)
    # Print success statistics
    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(
        success_grasp, all_grasp, success_grasp/all_grasp))     
    print('\n>>>>>>>>>>>>>>>>>>>> Percent Cleared: {}/{}={}'.format(
        success_grasp, objs_num, success_grasp/objs_num))    
