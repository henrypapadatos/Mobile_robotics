# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:04:55 2021

@author: henry papapatos
"""

import pyvisgraph as vg
import numpy as np
import cv2
import math
import mlrose


def draw(image, start, goal, obstacle):
    cv2.circle(image, start,radius=0, color=(0,255,0), thickness=5)

    for current_goal in goal:
        cv2.circle(image, current_goal,radius=0, color=(0,255,255), thickness=5) 

    #draw obstacle
    cv2.fillPoly(image, obstacle, color=(0,0,255))
    
 
def Astar(start, goal, obstacle,draw = False, image = None):
    polys = []
    for cobstacle in obstacle:
        cpolys = []
        for cpos in cobstacle:
            cpolys.append(vg.Point(cpos[0],cpos[1]))
        polys.append(cpolys)
    
    point = goal.copy()
    point.insert(0, start)
    dist_list=[]
    path_list=[]

    g = vg.VisGraph()
    g.build(polys)
        
    for i in range(len(point)):
        for k in range(len(point)-i-1):
            j = k+i+1
            shortest = g.shortest_path(vg.Point(point[i][0], point[i][1]), vg.Point(point[j][0], point[j][1]))
            path_list.append([i, j, shortest])
            dist_list.append((i, j, path_length(shortest)))
            
            if draw:
                h = 0
                for h in range (len(shortest)-1):
                    dstart = np.array([int(shortest[h].x), int(shortest[h].y)])
                    dend = np.array([int(shortest[h+1].x), int(shortest[h+1].y)])
                    image = cv2.line(image,dstart, dend, color=(0,255,0), thickness = 1)
                
    return dist_list, path_list

def path_length(path):
    
    distance = 0
    for i in range(len(path)-1):
        distance += math.sqrt((path[i].x-path[i+1].x)**2 + (path[i].y-path[i+1].y)**2)
    
    return distance

def tsp(dist_list, path_list, length, draw = False, image = None):
    
    optimal_trajectory=[]

    # Initialize fitness function object using dist_list
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
    
    dist_list, path_list = Astar(start, goal, obstacle)
    
    optimal_path = tsp(dist_list, path_list, len(goal)+1, draw, image)
    
    for i in range(len(optimal_path)):
        optimal_path[i] = optimal_path[i] * conversion_factor
        
    return optimal_path