#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import time
import math

import numpy as np

# --------------------------------------Secondary Functions-----------------------------------------------

def centroid(vertexes):
    x_list = [vertex [0][0] for vertex in vertexes]
    y_list = [vertex [0][1] for vertex in vertexes]
    length = len(vertexes)
    x = np.int0(sum(x_list) / length)
    y = np.int0(sum(y_list) / length)
    
    return(x, y)

# Expansion

def expand(centroid, vertexes, px_factor):
    
    half_thymio = 55*(1/px_factor) # Thymio's half width - converted from mm to pixels
    new_corners = []
    
    for vertex in vertexes:
        dist = [(vertex[0][0] - centroid[0]), (vertex[0][1] - centroid[1])]
        angle = np.arctan2(dist[1], dist[0])
        new_coord = [vertex[0][0] + np.cos(angle)*half_thymio, vertex[0][1] + np.sin(angle)*half_thymio]
        new_corners.append(new_coord)
    
    new_corners = np.int0(new_corners)
    return new_corners

def color_detect(pic, low, high):
    
    image=cv2.cvtColor(pic, cv2.COLOR_BGR2HSV) 
    mask=cv2.inRange(pic, low, high)
    image=cv2.blur(pic, (5, 5))
    mask=cv2.erode(mask, None, iterations=4)
    mask=cv2.dilate(mask, None, iterations=4)
    
    color_img =cv2.bitwise_and(pic, pic, mask=mask)
    
    return color_img, mask

def goals(pic):
    
    low_yellow = np.array([200,60,60])
    high_yellow = np.array([254,255,150])

    goals_loc = []
    
    # Extract goals from original image through color detection
    img_goals, mask_goals = color_detect(pic, low_yellow, high_yellow)

    # Extract centers of goals 
    contours=cv2.findContours(mask_goals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    for cont in contours:
        
        ((x, y), rayon)=cv2.minEnclosingCircle(cont)
        goals_loc.append((x,y))
    
    goals_loc = np.int0(goals_loc)
    
    #draw centers of goals
    for cent in goals_loc:
        cv2.circle(img_goals, cent, 5, (255, 0, 0) , -1)
    
    return goals_loc, img_goals

def obstacles(img):
    
    low_blue = np.array([45,50,70])
    high_blue = np.array([80,255,255])
    corners=[]
    new_corners=[]
    centroids=[]
    
    img_obst, mask_obst = color_detect(img, low_blue, high_blue)
    contours, hierarchy = cv2.findContours(mask_obst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=img2, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    for cont in contours:
        epsilon = 0.08 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        if(len(approx)>2):
            cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
            corners.append(approx)
            if(len(approx) == 3):
                Pix_to_mm = pix_to_mm(approx)
    
    # From extracted corners, define middle point of each object and create vertex (by 'expanding' corners)
    
    for i in range(0, len(corners)):
        centroids.append(centroid(corners[i]))
        #cv2.circle(poly_copy, centroids[i], 5, (255, 0, 0) , -1)# Just to check if the centroids are good
        new_corners.append(expand(centroids[i], corners[i], Pix_to_mm))   # Determine expanded corners (to take into account thymio width)    
    
        for corn in new_corners[i]:
            cv2.circle(img_obst, corn, 3, (255, 0, 0) , -1)

    return new_corners, Pix_to_mm, img_obst

def pix_to_mm(triangle):
    
    len_in_px = 0
    len_in_mm = 30 # To define...
    
    for corn1 in triangle:
        for corn2 in triangle:
            if(np.any(corn1 != corn2)):
                dist = math.dist(corn1[0], corn2[0])
                if(dist > len_in_px):
                    len_in_px = dist

    px_to_mm = len_in_mm/len_in_px
    
    return px_to_mm

def start(img):
    
    low_red = np.array([115,20,20]) 
    high_red = np.array([180,100,150])
    
    corners = [(0,0), (0,0)]
    centers = []
    start = []
    
    img_angle, mask_angle = color_detect(img, low_red, high_red)
    contours, hierarchy = cv2.findContours(mask_angle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in contours:
        
        epsilon = 0.1 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        
        if(len(approx)==4):
            start = centroid(approx)
    
    return start
    
    
# --------------------------------------Main Functions-----------------------------------------------
    
def Init(image):
     
    start_pos = start(image)
    vertexes, px_to_mm, img_obst = obstacles(image)
    goals_pos, img_goals = goals(image)
    
    return start_pos, vertexes, goals_pos, px_to_mm

def vision(image, px_factor):
    
    # Determine pose of robot based on two simple red shapes on its top
    # One shape is enough for position but a second one is needed to determine the angle
    
    # HSV code for the red used
    low_red = np.array([115,20,20]) 
    high_red = np.array([180,100,150])
    
    corners = [(0,0), (0,0)]
    centers = []
    pose_hidden = np.array([0,0,0])
    hidden = False
    
    # Extract the red from the img given by the camera 
    img_angle, mask_angle = color_detect(image, low_red, high_red)
    contours, hierarchy = cv2.findContours(mask_angle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    # Hidden camera condition
    if len(contours) == 0: 
        print('No contours found for goals')
        hidden = True
        return pose_hidden, hidden
   
    # From contours of red shapes, approximate polygons and obtain corners
    for cont in contours:
        
        epsilon = 0.1 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        
        if(len(approx)>2):
            if(len(approx)==3):
                corners[0] = approx
            if(len(approx)==4):
                corners[1] = approx
    
    # From corners, get centers
    for i in range(0, len(corners)):
        if(len(corners[i]) == 4):
            center_rect = centroid(corners[i])
            centers.append(center_rect)
            (x,y) = center_rect # Center of the robot is the center of the red rectangle
        else:
            centers.append(centroid(corners[i]))

    diff = [centers[0][0]-centers[1][0], centers[0][1]-centers[1][1]] # Distance between the two shapes 
    angle = math.degrees(np.arctan2(diff[1], diff[0])%(2*np.pi))
    
    pose = np.array([x*px_factor, y*px_factor, angle]) # Return pose of robot as an array
    
    return pose, hidden

