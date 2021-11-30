# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:03:23 2021

@author: henry papapatos
"""
import numpy as np
import filters 
import math
import cv2 
import computer_vision

TRESH_DIST = 3 #mm
CAMERA = 1 # Camera

def main():
                     
    # The estimated state vector at time t-1 in the global reference frame.
    # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
    
    cap=cv2.VideoCapture(CAMERA) # Either 0 or 1, front camera or external cam

    # Get 1st frame of video 
    ret, frame_init = cap.read()
    #width, height = frame_init.shape[1], frame1.shape[0]
    
    start_pos, vertexes, goals_pos, px_to_mm = computer_vision.Init(frame_init)
    
    goal_list = [] # Path planning
    
    ###### APPELER get_vision_position##########
    x_est_t_minus_1, hidden = computer_vision.vision(frame_init, px_to_mm)
     
    # The control input vector at time t-1 in the global reference frame.
    # [v, yaw_rate]
    u_t_minus_1 = np.array([0 ,0])
     
    # State covariance matrix P_t_minus_1
    P_t_minus_1 = np.array([[0.1,  0,   0],
                            [  0,0.1,   0],
                            [  0,  0, 0.1]])

# Extract vertexes, goals, thymio's start position and orientation from first frame 
# .
# .
# . 
    

    while len(goal_list)!=0:
        
        ret, frame = cap.read()
        
        ###### APPELER get_vision_position##########
        obs_vector_z_t, hidden = computer_vision.vision(frame, px_to_mm)
        
        print(f'Timestep measurement={obs_vector_z_t}')
        
        #######ADD TIME MEASURE TWO KALMAN CALLS#######
        dt = 0 ; #A voir a modifier, Aitana
        
        ####### ADD HIDDEN CONDITION !!!!#############
        # Run the Extended Kalman Filter and store the 
        # near-optimal state and covariance estimates
        optimal_state_estimate_t, covariance_estimate_t = filters.ekf(
            obs_vector_z_t, # Most recent sensor measurement
            x_est_t_minus_1, # Our most recent estimate of the state
            u_t_minus_1, # Our most recent control input
            P_t_minus_1, # Our most recent state covariance matrix
            dt) # Time interval
        
        if np.linalg.norm(optimal_state_estimate_t-goal_list[0]) < TRESH_DIST:
            goal_list.pop(0)
            
        # Get ready for the next timestep by updating the variable values
        x_est_t_minus_1 = optimal_state_estimate_t
        P_t_minus_1 = covariance_estimate_t
        
        ######APPELER PID##########
        u_t_minus_1 = 0; # delta_v returned by PID
        
        # Print a blank line
        print()
    
    cap.release()
    cv2.destroyAllWindows()
 
main()
