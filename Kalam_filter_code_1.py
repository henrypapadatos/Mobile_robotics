import numpy as np
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
        control_vector_k_minus_1, P_k_minus_1, dk):
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
    state_estimate_k = A_k_minus_1 @ (
            state_estimate_k_minus_1) + (
            getB(state_estimate_k_minus_1[2],dk)) @ (
            control_vector_k_minus_1) + (
            process_noise_v_k_minus_1)
             
    print(f'X_est State Estimate Before EKF={state_estimate_k}')
             
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
 
    print(f'Z Measurements={z_k_observation_vector}')
             
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
    print(f'X_est State Estimate After EKF={state_estimate_k}')
 
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
     
def main():

    '''
    #odometry
    thymio_v_to_mms = 0.34 # (0.3436426116838488 ou  0.333333333)
    thymio_y_to_mms = 0.06085878507832751
    offset = 5
    
    v_right = thymio_v_to_mms * v_right
    v_left = thymio_v_to_mms *v_left - offset
    
    v = (v_right + v_left)/2
    yaw_rate = (v_right - v_left)*thymio_y_to_mms
    '''
 
    # We start at time t=1
    t = 1
                     
    # The estimated state vector at time t-1 in the global reference frame.
    # [x_t_minus_1, y_t_minus_1, yaw_t_minus_1]
    # [meters, meters, radians]
    x_est_t_minus_1 = np.array([0.0,0.0,0.0])
     
    # The control input vector at time t-1 in the global reference frame.
    # [v, yaw_rate]
    # [meters/second, radians/second]
    # In the literature, this is commonly u.
    # Because there is no angular velocity and the robot begins at the 
    # origin with a 0 radians yaw angle, this robot is traveling along 
    # the positive x-axis in the global reference frame.
    u_t_minus_1 = np.array([0 ,0])
     
    # State covariance matrix P_t_minus_1
    # This matrix has the same number of rows (and columns) as the 
    # number of states (i.e. 3x3 matrix). P is sometimes referred
    # to as Sigma in the literature. It represents an estimate of 
    # the accuracy of the state estimate at time t made using the
    # state transition matrix. We start off with guessed values.
    P_t_minus_1 = np.array([[0.1,  0,   0],
                            [  0,0.1,   0],
                            [  0,  0, 0.1]])
                             
    # Start at t=1 and go through each of the 5 sensor observations, 
    # one at a time. 
    # We stop right after timestep t=5 (i.e. the last sensor observation)
    while (goal_list!=0)
        
        # Print the current timestep
        print(f'Timestep t={t}') 
        print(f'Timestep measurement={obs_vector_z_t}')
         
        # Run the Extended Kalman Filter and store the 
        # near-optimal state and covariance estimates
        optimal_state_estimate_t, covariance_estimate_t = ekf(
            obs_vector_z_t, # Most recent sensor measurement
            x_est_t_minus_1, # Our most recent estimate of the state
            u_t_minus_1, # Our most recent control input
            P_t_minus_1, # Our most recent state covariance matrix
            dt) # Time interval
         
        # Get ready for the next timestep by updating the variable values
        x_est_t_minus_1 = optimal_state_estimate_t
        P_t_minus_1 = covariance_estimate_t
         
        # Print a blank line
        print()
 
# Program starts running here with the main method  

    
    
    
    
    