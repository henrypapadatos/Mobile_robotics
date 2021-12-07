# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:04:19 2021

@author: Juliette
"""


def obstacle_avoidance(horz_prox):
    w_l = [40,  20, -20, -20, -40,  30, -10, 8, 0]
    w_r = [-40, -20, -20,  20,  40, -10, 30, 0, 8]

    # Scale factors for sensors and constant factor
    sensor_scale = 200
    constant_scale = 20
    
    x = [0,0,0,0,0,0,0,0,0]
    
    y=[0,0]
   
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
        
    return y


    

