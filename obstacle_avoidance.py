"""
Created on Tue Dec  7 10:04:19 2021

@author: Juliette
"""
SENS_SCALE = 500
CST_SCALE = 60
SPEED_TRH = 30
AVG_SPEED = 40


def obstacle_avoidance(horz_prox, speedl, speedr):
    w_l = [30,  20, -20, -20, -30,  0, 0, 6, 0]
    w_r = [-30, -20, -20,  20,  30, 0, 0, 0, 6]

    # Scale factors for sensors and constant factor
    sensor_scale = SENS_SCALE
    constant_scale = CST_SCALE
    Thr = SPEED_TRH
    
    x = [0,0,0,0,0,0,0,0,0]
    
    y = [speedl,speedr]
   
    x[7] = y[0]//constant_scale
    x[8] = y[1]//constant_scale

    for i in range(7):
        # Get and scale inputs
        x[i] = horz_prox[i] // sensor_scale

    y = [0,0]   

    for i in range(len(x)):    
        # Compute outputs of neurons and set motor powers
        y[0] = y[0] + x[i] * w_l[i]
        y[1] = y[1] + x[i] * w_r[i]
        
    y[0] = y[0]+AVG_SPEED  
    y[1] = y[1]+AVG_SPEED
        
    state = 1
    hiddencam = 1
    
    if y[0]<=Thr and y[1]<=Thr:
        state = 0
        hiddencam = 0
        
    return y, state, hiddencam