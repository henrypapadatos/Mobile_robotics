# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:27:10 2021

@author: papad
"""

import numpy as np

#define start position
start = np.array([5,5])

goal = []
goal.append(np.array([450,80]))
goal.append(np.array([300,300]))

obstacle = []
obstacle.append(np.array( [[10,50], [400,50], [90,200], [50,500]]))
obstacle.append(np.array( [[250,350], [400,500], [500,500]])) 

class point:
    def __init__(self, x, y, point_type):
        self.x = x
        self.y = y
        self.point_type = point_type
        self.neighbours = []
        
    def find_neighbours(self, obstacle_lines, other_points):
        #todo
        return
    
    def get_position(self):
        return np.array([self.x,self.y])
    