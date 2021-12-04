# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:03:23 2021

@author: henry papapatos
"""
import numpy as np
import sys
#sys.path.append(r'C:\Users\Usuario\OneDrive - epfl.ch\Documents\EPFL\Basics of mobile robotics\PROJET\Github\Mobile_robotics')
import math
import cv2 
import time
import filters 
import computer_vision
import path_planning	

TRESH_DIST = 3 #mm
CAMERA = 1 # Camera

def main():
                      
    cap=cv2.VideoCapture(CAMERA ) # Either 0 or 1, front camera or external cam

    # Get 1st frame of video 
    time.sleep(2)
    
    frame_init = computer_vision.get_image(cap)
    
    cv2.imshow('frame', frame_init)
    
    #cv2.waitKey(0)
    
    cv2.imwrite(r'C:\Users\papad\OneDrive\Images\Pellicule\img.jpg', frame_init) #POUR TUNING PAR ELIOTT
    
    # Extract vertexes, goals, thymio's start position and orientation from first frame
    start_pos, obst_vertexes, goals_pos, px_to_mm = computer_vision.Init(frame_init)
    
    computer_vision.display_obstacle(frame_init, start_pos, goals_pos, obst_vertexes)
    # computer_vision.display_pos(frame_init, pos, is_from_camera)
    cv2.imshow('frame', frame_init)
    
    cv2.waitKey(0)
    
    # goal_list = path_planning.get_optimal_path(start_pos, goals_pos, obst_vertexes) # Path planning
    
    # # The estimated state vector at time t-1 in the global reference frame.
    # # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
    # x_est_t_minus_1, cam_hidden = computer_vision.vision(frame_init, px_to_mm)
    # #x_est_t_minus_1 = np.array([0.0,0.0,0.0])
    
    # # The control input vector at time t-1 in the global reference frame.
    # # [v, yaw_rate]
    # u_t_minus_1 = np.array([0 ,0])
     
    # # State covariance matrix P_t_minus_1
    # P_t_minus_1 = np.array([[0.1,  0,   0],
    #                         [  0,0.1,   0],
    #                         [  0,  0, 0.1]])


    # #update le dt car sinon kalman marche pas
    # dt = 0 ; #A voir a modifier, Aitana
    
    # while len(goal_list)!=0:
        
    #     frame = = computer_vision.get_image(cap)
        
    #     ###### APPELER get_vision_position##########
    #     obs_vector_z_t, cam_hidden = computer_vision.vision(frame, px_to_mm)
        
    #     print(f'Timestep measurement={obs_vector_z_t}')

    #     # Run the Extended Kalman Filter and store the 
    #     # near-optimal state and covariance estimates
    #     if cam_hidden:
    #         #should I pudate covariance or not 
    #         _, _, optimal_state_estimate_t= filters.ekf(
    #             obs_vector_z_t, # Most recent sensor measurement
    #             x_est_t_minus_1, # Our most recent estimate of the state
    #             u_t_minus_1, # Our most recent control input
    #             P_t_minus_1, # Our most recent state covariance matrix
    #             dt) # Time interval
    #         print('Camera hidden')
    #     else: 
    #         optimal_state_estimate_t, covariance_estimate_t, _= filters.ekf(
    #             obs_vector_z_t, # Most recent sensor measurement
    #             x_est_t_minus_1, # Our most recent estimate of the state
    #             u_t_minus_1, # Our most recent control input
    #             P_t_minus_1, # Our most recent state covariance matrix
    #             dt) # Time interval
        
    #     if np.linalg.norm(optimal_state_estimate_t-goal_list[0]) < TRESH_DIST:
    #         goal_list.pop(0)
            
    #     # Get ready for the next timestep by updating the variable values
    #     x_est_t_minus_1 = optimal_state_estimate_t
    #     P_t_minus_1 = covariance_estimate_t
        
    #     #check if obstacle in coming
        
    #     #if no
    #     ######APPELER PID##########
    #     u_t_minus_1 = 0; # delta_v returned by PID
        
    #     #if yes, obstacle avoidance 
        
    #     # Print a blank line
    #     print()
    
    
 
try: 
    main()
finally:
    cap.release()
    cv2.destroyAllWindows()
