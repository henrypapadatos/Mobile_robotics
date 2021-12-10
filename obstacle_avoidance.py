"""
Created on Tue Dec  7 10:04:19 2021

@author: Juliette
"""
SENS_SCALE = 1300
CST_SCALE = 2
SPEED_TRH = 140
AVG_SPEED = 70


def obstacle_avoidance(horz_prox, speedl, speedr, verbose = False):
    
    w_l = [20,  20, -20, -20, -20,  0, 0, 1, 0]
    w_r = [-22, -22, -22,  22, 22, 0, 0, 0, 1]

    x = [0,0,0,0,0,0,0,0,0]
    
    y = [speedl,speedr]
   
    x[7] = y[0]/CST_SCALE
    x[8] = y[1]/CST_SCALE
    
    if verbose: print("remanent left speed: ", x[7])
    if verbose: print("remanent right speed: ", x[8])

    for i in range(7):
        # Get and scale inputs
        x[i] = horz_prox[i] / SENS_SCALE

    y = [0,0]   

    for i in range(len(x)):    
        # Compute outputs of neurons and set motor powers
        y[0] = y[0] + x[i] * w_l[i]
        y[1] = y[1] + x[i] * w_r[i]
        
    y[0] = int(y[0]+AVG_SPEED)
    y[1] = int(y[1]+AVG_SPEED)
    
    if verbose: print("left speed: ", x[7])
    if verbose: print("right speed: ", x[8])
        
    state = 1
    
    if y[0]<=SPEED_TRH and y[1]<=SPEED_TRH:
        state = 0
        
    return y, state