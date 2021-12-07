# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:23:37 2021

@author: Aitana Waelbroeck Boix
"""
import numpy as np
import math

MAX_SUM_ERROR = 50
KP_1 = 1
KI_1 = 0.05
KD_1 = 0
PI_CLOCK = 0.1
DEG_LIM = 65
SPEED_AVG = 150

# Description: Extended Kalman Filter example (two-wheeled mobile robot)

# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)
 
# A matrix
A_k_minus_1 = np.array([[1.0,  0,   0],
                       [  0, 1.0,   0],
                       [  0,   0, 1.0]])

# 
process_noise_v_k_minus_1 = np.array([0.01,0.01,0.003])
     
# State model noise covariance matrix Q_k
Q_k = np.array([[1.0,   0,   0],
                [  0, 1.0,   0],
                [  0,   0, 1.0]])
                 
# Measurement matrix H_k
H_k = np.array([[1.0,  0,   0],
                [  0,1.0,   0],
                [  0,  0, 1.0]])
                         
# Sensor measurement noise covariance matrix R_k
R_k = np.array([[1.0,   0,    0],
                [  0, 1.0,    0],
                [  0,    0, 1.0]])  
                 
# Sensor noise
sensor_noise_w_k = np.array([0.07,0.07,0.04])
 
def getB(yaw, deltak):
    """
    # Author: Addison Sears-Collins
    # https://automaticaddison.com
    Calculates and returns the B matrix
    3x2 matix -> number of states x number of control inputs
    The control inputs are the forward speed and the
    rotation rate around the z axis from the x-axis in the 
    counterclockwise direction.
    [v,yaw_rate]
    Expresses how the state of the system [x,y,yaw] changes
    from k-1 to k due to the control commands (i.e. control input).
    :param yaw: The yaw angle (rotation angle around the z axis) in rad 
    :param deltak: The change in time from time step k-1 to k in sec
    """
    B = np.array([  [np.cos(yaw)*deltak, 0],
                    [np.sin(yaw)*deltak, 0],
                    [0, deltak]])
    return B
 
def ekf(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk, hidden_cam, verbose = False):
    """
    # Author: Addison Sears-Collins
    # https://automaticaddison.com
    Extended Kalman Filter. Fuses noisy sensor measurement to 
    create an optimal estimate of the state of the robotic system.
         
    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            3x1 NumPy Array [v,v,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds
             
    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            3x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array                 
    """
    ######################### Predict #############################
    # Predict the state estimate at time k based on the state 
    # estimate at time k-1 and the control input applied at time k-1.
    if verbose: print(f'Timestep measurement={z_k_observation_vector}')
    state_estimate_k = A_k_minus_1 @ (
            state_estimate_k_minus_1) + (
            getB(state_estimate_k_minus_1[2],dk)) @ (
            control_vector_k_minus_1) + (
            process_noise_v_k_minus_1)
                    
    if verbose: print(f'X_est State Estimate Before EKF={state_estimate_k}')
    
    if not hidden_cam:
        # Predict the state covariance estimate based on the previous
        # covariance and some noise
        P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (
                Q_k)
             
        ################### Update (Correct) ##########################
        # Calculate the difference between the actual sensor measurements
        # at time k minus what the measurement model predicted 
        # the sensor measurements would be for the current timestep k.
        measurement_residual_y_k = z_k_observation_vector - (
                (H_k @ state_estimate_k) + (
                sensor_noise_w_k))
     
        if verbose: print(f'Z Measurements={z_k_observation_vector}')
                 
        # Calculate the measurement residual covariance
        S_k = H_k @ P_k @ H_k.T + R_k
             
        # Calculate the near-optimal Kalman gain
        # We use pseudoinverse since some of the matrices might be
        # non-square or singular.
        K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
             
        # Calculate an updated state estimate for time k
        state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
         
        # Update the state covariance estimate for time k
        P_k = P_k - (K_k @ H_k @ P_k)
         
        # Print the best (near-optimal) estimate of the current state of the robot
        if verbose: print(f'X_est State Estimate After EKF={state_estimate_k}')
    else:
        if verbose: print('THE CAMERA IS HIDDEN')
        P_k = P_k_minus_1
        
    if verbose: print(f'covariance_estimate_t={P_k}')
    if verbose: print()
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k

def pid(pos_robot, goal_pos, sum_error, alt_error_pid,dt, verbose = False):
    print("pos robot is: ", pos_robot)
    
    if verbose: print("y:", pos_robot[1]-goal_pos[1])
    if verbose: print("x:", pos_robot[0]-goal_pos[0])
    
    angle_rad = np.arctan2(pos_robot[1]-goal_pos[1],pos_robot[0]-goal_pos[0])

    if verbose: print("Robot_angle:",360-pos_robot[2])
    
    #calculer le delta error angle
    error = 360-(pos_robot[2]+math.degrees(angle_rad%(2*np.pi)))
    
    if verbose: print("Error_angle:",error)
    
    #disables the PID regulator if the error is to small 
    #disables the PID regulator if the error is to small
    #this avoids to always move as we cannot exactly be where we want and
    #the camera is a bit noisy
    # if abs(error) < ERROR_THRESHOLD:
    #     return 0
    
    sum_error = sum_error + error
    
    #we set a maximum and a minimum for the sum to avoid an uncontrolled growth
    if sum_error > MAX_SUM_ERROR:
        sum_error = MAX_SUM_ERROR
    elif sum_error < -MAX_SUM_ERROR:
        sum_error = -MAX_SUM_ERROR
    	
    
    speed = KP_1 * error + KI_1 * sum_error + KD_1 * ((error - alt_error_pid)/dt)
    
    if error > DEG_LIM:
        v_l = speed
        v_d= -speed
        if verbose: print("Turning only, speed l and r:",v_l,v_d)
    else:
        v_l = SPEED_AVG + speed
        v_d = SPEED_AVG -speed
        if verbose: print("Moving and turning, speed l and r:",v_l,v_d)
    
    if verbose: print("Sum_error, Alt_error_pid:", sum_error,alt_error_pid)
    
    return sum_error, error, v_l, v_d


###### How to call PID ######

# def main ():
#     sum_error = 0 
#     alt_error_pid = 0 
#     pos_robot= [2,1,300]
#     goal_pos = [6,3]

#     speed,sum_error, alt_error_pid, v_l,v_d = pid(pos_robot,goal_pos,sum_error, alt_error_pid)

# main()

###############################
