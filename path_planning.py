# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:04:55 2021

@author: henry papapatos
"""

import pyvisgraph as vg
import numpy as np
import cv2
import math
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
    
def visibility_graph(start, goal, obstacle,draw = False, image = None):
    """
    Use visibility graph and Djikstra algorithm to find the shortest
    and its corresponding path between each pair of goals.
    This library is used: https://github.com/TaipanRex/pyvisgraph
    
    param start: position of the robot at the beginning
    param goal: list of position of goals where the robot has to go
    param obstacle: list of position of the corners of the extended obstacles
    param draw: if True, all the shortest path between pair of goals/start will be ploted
    param image: the image on which we draw the path
    
    return dist_list: list of the lengths of all shortest paths
    return path_list: list of intermediate coordinates composing each of shortest path
    """
    
    polys = [] 
    #creates the list of obstacle with the format needed by pyvisgraph library
    for cobstacle in obstacle: 
        cpolys = []
        for cpos in cobstacle:
            cpolys.append(vg.Point(cpos[0],cpos[1])) 
        polys.append(cpolys)
    
    #point is the list of goals in wich we insert the start position
    point = goal.copy()
    point.insert(0, start)
    dist_list=[]
    path_list=[]

    #create the visibility graph
    g = vg.VisGraph()
    g.build(polys)
        
    #for each pair of point (in point list), we compute the shortest path
    for i in range(len(point)):
        for k in range(len(point)-i-1):
            j = k+i+1
            #computes the shortest path between point i and j
            shortest = g.shortest_path(vg.Point(point[i][0], point[i][1]), vg.Point(point[j][0], point[j][1]))
            #add the shortest path into the path_list
            path_list.append([i, j, shortest])
            #add the length of the sortest path in the dist_list
            dist_list.append((i, j, path_length(shortest)))
            
            #if param draw = True, draw all the sortest paths on param image
            if draw:
                h = 0
                for h in range (len(shortest)-1):
                    dstart = np.array([int(shortest[h].x), int(shortest[h].y)])
                    dend = np.array([int(shortest[h+1].x), int(shortest[h+1].y)])
                    image = cv2.line(image,dstart, dend, color=(0,255,0), thickness = 1)
                
    return dist_list, path_list

def path_length(path):
    """
    Computes the length of a path
    
    param path: positions of all the points composing the path
    
    return distance: length of the path
    """
    distance = 0
    for i in range(len(path)-1):
        distance += math.sqrt((path[i].x-path[i+1].x)**2 + (path[i].y-path[i+1].y)**2)
    
    return distance

def tsp(dist_list, path_list, length, draw = False, image = None):
    """
    Solve the travelling salesman problem using a genetic algorithm.
    The library used is mlrose: https://github.com/gkhayes/mlrose
    
    param dist_list: list of the lengths of all shortest paths
    param path_list: list of intermediate coordinates composing each of shortest path
    param length: number of points to travel to
    param draw: if True, all the shortest path between pair of goals/start will be ploted
    param image: the image on which we draw the path
    
    return optimal_trajectory: list of points that compose the optimal path (expressed in pixel coordinates)
    """
    
    optimal_trajectory=[]

    # Initialize fitness function object using dist_list (dist_list is the length of the 
    # pairwise shortest path computed in visibility_graph function)
    fitness_dists = mlrose.TravellingSales(distances = dist_list)
    # Define optimization problem object
    problem_fit2 = mlrose.TSPOpt(length, fitness_fn = fitness_dists, maximize = False)
    # Solve using genetic algorithm
    best_state, best_fitness = mlrose.genetic_alg(problem_fit2, mutation_prob = 0.2, max_attempts = 100,
                                                  random_state = 2)
    
    #rearrange best state so that the path starts at the start position
    while best_state[0] != 0:
        best_state = np.roll(best_state,1)
        
        
    #fill the optimal tarjectory list using best_state and path_list
    for i in range(len(best_state)):
        start = best_state[i]
        if i == len(best_state)-1:
            stop = best_state[0]
        else:
            stop = best_state[i+1]
            
        for segment in path_list:
            if (segment[0] == start and segment[1] == stop):
                segment_path = segment[2]
                for j in range(len(segment_path)-1):
                    optimal_trajectory.append(np.array([segment_path[j].x, segment_path[j].y]))
            if (segment[1] == start and segment[0] == stop):
                segment_path = segment[2]
                for k in range(len(segment_path)-1):
                    j = len(segment_path) - 1 - k
                    optimal_trajectory.append(np.array([segment_path[j].x, segment_path[j].y]))
        
    optimal_trajectory.append(np.array([path_list[0][2][0].x,path_list[0][2][0].y]))  
    
    #if param draw = True, draw the sortest path on param image
    if draw:
        for i in range(len(optimal_trajectory)):
            start = optimal_trajectory[i].astype(int)
            if i == len(optimal_trajectory)-1:
                stop = optimal_trajectory[0].astype(int)
            else:
                stop = optimal_trajectory[i+1].astype(int)
                
            image = cv2.line(image,start, stop, color=(255, 153, 255), thickness = 1)
    
   
    return optimal_trajectory

def get_optimal_path(start, goal, obstacle,conversion_factor = 1, draw = False, image = None):
    """
    compute the optimal path to join all the goals and then go back to the start position 
    while avoiding the obstacles
    
    param start: position of the robot at the beginning
    param goal: list of position of goals where the robot has to go
    param obstacle: list of position of the corners of the extended obstacles
    param conversion_factor: ratio to convert pixels in mm
    param draw: if True, all the shortest path between pair of goals/start will be ploted
    param image: the image on which we draw the path
    
    return optimal_trajectory: list of points that compose the optimal path (expressed in mm coordinates)
    """
    dist_list, path_list = visibility_graph(start, goal, obstacle)
    
    optimal_path = tsp(dist_list, path_list, len(goal)+1, draw, image)
    
    #converts the position from pixel to mm
    for i in range(len(optimal_path)):
        optimal_path[i] = optimal_path[i] * conversion_factor
        
    return optimal_path