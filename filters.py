# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:23:37 2021

@author: Aitana Waelbroeck Boix

"""
import numpy as np
import math

KP = 2
DEG_LIM = 10
SPEED_AVG = 150

# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)
                  
def getB(yaw, deltak):
    """
    Calculates and returns the B matrix.
    The control inputs are the forward speed and the
    rotation rate around the z axis.[v,yaw_rate]
    Expresses how the state of the system changes
    from k-1 to k due to the control commands (i.e. control input). 
    """
    B = np.array([  [np.cos(np.deg2rad(yaw))*deltak, 0],
                    [np.sin(np.deg2rad(yaw))*deltak, 0],
                    [0, deltak]])
    return B
 
def kf(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk, hidden_cam, verbose = False):
    """
    Modified from Addison Sears-Collins: https://automaticaddison.com
    
    Kalman Filter. Fuses noisy sensor measurement to 
    create an optimal estimate of the state of the robotic system.
         
    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [mm,mm,degrees].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [mm,mm,degrees].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            3x1 NumPy Array [v,v,yaw rate] in the global reference frame
            in [mm/s,mm/s,degrees/s)].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds
             
    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            3x1 NumPy Array ---> [mm,mm,degrees]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array                 
    """
    # A matrix
    A_k_minus_1 = np.array([[1.0,  0,   0],
                           [  0, 1.0,   0],
                           [  0,   0, 1.0]])

    # State model noise covariance matrix Q_k
    Q_k = np.array([[1.0,   0,   0],
                    [  0, 1.0,   0],
                    [  0,   0, 1.0]])
                     
    # Measurement matrix H_k
    H_k = np.array([[1.0,  0,   0],
                    [  0,1.0,   0],
                    [  0,  0, 1.0]])
                             
    # Sensor measurement noise covariance matrix R_k
    R_k = np.array([[0.1,   0,    0],
                    [  0, 0.1,    0],
                    [  0,    0, 0.1]])  
    
    ######################### Predict #############################
    # Predict the state estimate at time k based on the state 
    # estimate at time k-1 and the control input applied at time k-1.
    if verbose: print(f'Timestep measurement={z_k_observation_vector}')

    state_estimate_k = A_k_minus_1 @ (
            state_estimate_k_minus_1) + (
            getB(state_estimate_k_minus_1[2],dk)) @ (
            control_vector_k_minus_1)
                    
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
                H_k @ state_estimate_k)
     
        if verbose: print(f'Z Measurements={z_k_observation_vector}')
                 
        # Calculate the measurement residual covariance
        S_k = H_k @ P_k @ H_k.T + R_k
             
        # Calculate the Kalman gain
        K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
             
        # Calculate an updated state estimate for time k
        state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
         
        # Update the state covariance estimate for time k
        P_k = P_k - (K_k @ H_k @ P_k)
         
        # Print the best estimate of the current state of the robot
        if verbose: print(f'X_est State Estimate After EKF={state_estimate_k}')
    else:
        if verbose: print('THE CAMERA IS HIDDEN')
        P_k = P_k_minus_1
        
        
    if verbose: print(f'covariance_estimate_t={P_k}')
    if verbose: print()
    
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k


def p_controler(pos_robot, pos_goal, verbose = False):
    '''
    Computes the speed of the robot proportionaly to the error orientation
    /remaining distance to the goal.
    
    INPUT
        :param pos_robot 
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [mm,mm,degrees].
        :param pos_goal 
            3x1 NumPy Array [x,y] in the global reference frame
            in [mm,mm].
        :param dt Time interval in seconds
             
    OUTPUT
        :return v_l Left wheel speed: mm/s
        :return v_r Right wheel speed: mm/s
    '''
    if verbose: print("y:", pos_goal[1]-pos_robot[1])
    if verbose: print("x:", pos_goal[0]-pos_robot[0])
    
    angle_rad = np.arctan2(pos_goal[1]-pos_robot[1],pos_goal[0]-pos_robot[0])
    
    angle_robot = twopi_to_pi(pos_robot[2])

    if verbose: print("Robot_angle:",angle_robot)
    
    angle_goal = twopi_to_pi(math.degrees(angle_rad%(2*np.pi)))
        
    # Computes the error angle from the angle of the robot in the global 
    # reference frame and the angle between the robot and the goal.
    error = twopi_to_pi(angle_robot-angle_goal)
    
    if verbose: print("Robot to goal angle: ", angle_goal)
    if verbose: print("Error_angle:",error)
    
    # Computes a speed proportional to the error
    speed = KP * error
    
    # Only moves forward while turning if the error angle is really small
    if abs(error) > DEG_LIM:
        v_l = - speed
        v_r = speed
        if verbose: print("Turning only, speed l and r:",v_l,v_r)
    else:
        v_l = SPEED_AVG - speed
        v_r = SPEED_AVG + speed
        if verbose: print("Moving and turning, speed l and r:",v_l,v_r)
    
    return v_l, v_r


def twopi_to_pi(angle):
    '''
    Converts 0-pi angles to 0-2pi
    '''
    if angle < -180:
        angle = 360 + angle
    if angle >  180:
        angle = angle - 360
    
    return angle

