#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import time
import math

import numpy as np

LEN_IN_MM = 113 # lenght of one side of the cube in mm
SAFETY_FACTOR = 55 # margin so that the robot doesn't hit obstacles 
POLY_FACTOR_OBST = 0.05 # factor that determines how accurately the approxPolyDP function approximates
POLY_FACTOR_ROB = 0.05

GREEN_LOW_H = 50
GREEN_HIGH_H = 70
GREEN_LOW_S = 50
GREEN_HIGH_S = 255
GREEN_LOW_V = 50
GREEN_HIGH_V = 255

BLUE_LOW_H = 0
BLUE_HIGH_H = 20
BLUE_LOW_S = 50
BLUE_HIGH_S = 255
BLUE_LOW_V = 0
BLUE_HIGH_V = 255

RED_LOW_H = 115
RED_HIGH_H = 130
RED_LOW_S = 50
RED_HIGH_S = 255
RED_LOW_V = 50
RED_HIGH_V = 255

X_CROP_LEFT = 200
X_CROP_RIGHT = 1400
Y_CROP_TOP = 0
Y_CROP_BOT = 1080

# --------------------------------------Secondary Functions-----------------------------------------------

def centroid(vertexes):
    # Computes the coordinates of the centroid of a polygon given its corners coordinates
    
    # param vertexes : corners coordinates of polygon
    
    # return (x,y) : tuple of coordinates of the centroid of the polygon
    
    x_list = [vertex [0][0] for vertex in vertexes]
    y_list = [vertex [0][1] for vertex in vertexes]
    length = len(vertexes)
    x = np.int0(sum(x_list) / length)
    y = np.int0(sum(y_list) / length)
    
    return(x, y)

def expand(centroid, vertexes, px_factor):
    # Computes the expanded vertexes of a polygon based on the coordinates of its corners and centroid
    
    # param centroid : coordinates of centroid of obstacle 
    # param vertexes : corners coordinates of obstacle
    # param px_factor : pixel to millimeter conversion factor
    
    # return new_corners : coordinates of expanded corners
    
    
    half_thymio = SAFETY_FACTOR*(1/px_factor) # Thymio's half width - converted from mm to pixels
    if(len(vertexes) == 3): 
        expansion_dist = half_thymio/np.cos(np.pi/3) # For triangular obstacles
        
    else:
        expansion_dist = math.sqrt(2*half_thymio**2) # For rectangular obstacles
        
        
    new_corners = []
    
    for vertex in vertexes:
        dist = [(vertex[0][0] - centroid[0]), (vertex[0][1] - centroid[1])] # Distance between centroid and corner
        angle = np.arctan2(dist[1], dist[0])                                # Angle between centroid and corner
        new_coord = [vertex[0][0] + np.cos(angle)*expansion_dist, vertex[0][1] + np.sin(angle)*expansion_dist]
        new_corners.append(new_coord)
    
    new_corners = np.int0(new_corners)
    return new_corners

def color_detect(pic, low, high):
    # Extract color from the image "pic" based on the HSV color range [low-high]
    
    # param pic : image on which the color detection is conducted
    # param low : contains the lower values of the HSV parameters range
    # param high : contains the higher values of the HSV parameters range
    
    # return color_img : original image with black pixels except for the color detected
    # return mask : black and white color filter
    
    sigma = (5,5)
    
    image=cv2.blur(pic, sigma)                 # Blurring to get rid of image noise
    image=cv2.cvtColor(pic, cv2.COLOR_RGB2HSV) 
    mask=cv2.inRange(image, low, high)
    mask=cv2.erode(mask, None, iterations=4)    # Processing to have smoother color filter
    mask=cv2.dilate(mask, None, iterations=4)   

    color_img =cv2.bitwise_and(pic, pic, mask=mask)
    
    return color_img, mask

def goals(pic):
    # Extract goals from image "pic" and find their centers
    
    # param pic : image captured by the camera
    
    # return goals_loc : coordinates of the centers of the goals
    # return img_goals : original image with black pixels except for the goals
    
    low_green = np.array([GREEN_LOW_H, GREEN_LOW_S, GREEN_LOW_V])
    high_green = np.array([GREEN_HIGH_H, GREEN_HIGH_S, GREEN_HIGH_V])

    goals_loc = []
    
    # Extract goals from original image through color detection
    img_goals, mask_goals = color_detect(pic, low_green, high_green)

    contours=cv2.findContours(mask_goals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # For each contour found, fit the minimum enclosing circle and extract center
    for cont in contours: 
        
        ((x, y), rayon)=cv2.minEnclosingCircle(cont)
        goals_loc.append(np.array([int(x),int(y)]))
    
    return goals_loc, img_goals

def obstacles(img):
    # Extract obstacles from image "img", approximate them as polygons, find their corners and expand them
    
    # param pic : image captured by the camera
    
    # return new_corners : coordinates of the expanded obstacles corners
    # return Pix_to_mm : pixel to millimeter conversion factor
    # return img_goals : original image with black pixels except for the obstacles
    
    low_blue = np.array([BLUE_LOW_H, BLUE_LOW_S, BLUE_LOW_V])
    high_blue = np.array([BLUE_HIGH_H, BLUE_HIGH_S, BLUE_HIGH_V])
    corners=[]
    new_corners=[]
    centroids=[]
    
    # Extract obstacles from original image through color detection
    img_obst, mask_obst = color_detect(img, low_blue, high_blue)
    contours, hierarchy = cv2.findContours(mask_obst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest area for noise condition
    areas = [cv2.contourArea(c) for c in contours]
    
    if not areas:
        cv2.imshow('mask', mask_obst)
        cv2.waitKey(0)
        raise ValueError("Can not read frame")
        
    max_cont = max(areas)
    
    # For each contour found, approximate it as a polygon and extract its corners
    for cont in contours:
        epsilon = POLY_FACTOR_OBST * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        
        if(len(approx)>2 and cv2.contourArea(approx) >= max_cont/3): # Condition to get rid of detected noise 
            corners.append(approx)
            if(len(approx) == 4): # Use rectangle obstacle to find the pixel to millimeter conversion factor
                Pix_to_mm = pix_to_mm(approx)
    
    # From extracted corners, define middle point of each object and create vertex (by 'expanding' corners)
    
    for i in range(0, len(corners)):
        centroids.append(centroid(corners[i]))
        new_corners.append(expand(centroids[i], corners[i], Pix_to_mm))   # Determine expanded corners (to take into account thymio width)    

    return new_corners, Pix_to_mm, img_obst

def pix_to_mm(rectangle):
    # Extract longest side of rectangle and computes pixel to millimeter conversion factor from it
    
    # param rectangle : coordinates of the corners of the rectangular obstacle
    
    # return px_to_mm : pixel to millimeter conversion factor
    
    len_in_px = 0
    
    for corn1 in rectangle:
        for corn2 in rectangle:
            if(np.any(corn1 != corn2)):
                dist = math.dist(corn1[0], corn2[0])
                if(dist > len_in_px):
                    len_in_px = dist

    px_to_mm = LEN_IN_MM/len_in_px
    
    return px_to_mm

def start(img):
    # Extract red shapes on Thymio from image "img" and computes the initial position of the Thymio

    # param pic : first image captured by the camera
    
    # return start : coordinates of the starting position of the Thymio
    
    low_red = np.array([RED_LOW_H, RED_LOW_S, RED_LOW_V]) 
    high_red = np.array([RED_HIGH_H, RED_HIGH_S, RED_HIGH_V])
    
    corners = [(0,0), (0,0)]
    centers = []
    start = []
    
    # Extract obstacles from original image through color detection
    img_angle, mask_angle = color_detect(img, low_red, high_red)
    contours, hierarchy = cv2.findContours(mask_angle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest area for noise condition    
    areas = [cv2.contourArea(c) for c in contours]
    
    if not areas:
        cv2.imshow('mask', mask_angle)
        cv2.waitKey(0)
        raise ValueError("Can not read frame")
    
    max_cont = max(areas)
    
    # For each contour found, approximate it as a polygon and, if it is the red rectangle, extract its centroid
    for cont in contours:
        
        epsilon = POLY_FACTOR_ROB * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        
        if(len(approx)==4 and cv2.contourArea(approx) >= max_cont/2):
            start = np.array(centroid(approx))
    
    return start
    
    
# --------------------------------------Main Functions-----------------------------------------------
# Functions called in main
    
def Init(image):
    # Initialise program by analysing environment: compute starting position of Thymio, expanded vertexes of obstacles
    # pixel to millimeter conversion factor and centers of goals
    
    # param image : first image captured by camera
    
    # return start_pos : coordinates of the starting position of the Thymio
    # return vertexes : coordinates of the expanded obstacles corners
    # return goals_pos : coordinates of the centers of the goals
    # return px_to_mm : pixel to millimeter conversion factor
    
    start_pos = start(image)
    vertexes, px_to_mm, img_obst = obstacles(image)
    goals_pos, img_goals = goals(image)
    
    return start_pos, vertexes, goals_pos, px_to_mm

def vision(image, px_factor):
    # Determine pose of robot based on two simple red shapes on its top
    # One shape is enough for position but a second one is needed to determine the angle
    
    # param image : image captured by the camera
    # param px_factor : pixel to millimeter conversion factor
    
    # return pose : numpy array containing the coordinates of the position of the Thymio as well as its orientation
    # return hidden : boolean which is False when the camera is not hidden and True when hidden (Kalman filter adapts consequently)
    # return mask_angle : original image with black pixels except for the red shapes on top of the Thymio
    
    # HSV code for the red color
    low_red = np.array([RED_LOW_H, RED_LOW_S, RED_LOW_V]) 
    high_red = np.array([RED_HIGH_H, RED_HIGH_S, RED_HIGH_V])
    
    corners = [(0,0), (0,0)]
    centers = []
    pose_hidden = np.array([0,0,0])
    hidden = False
    
    # Extract red from the original image through color detection
    img_angle, mask_angle = color_detect(image, low_red, high_red)
    
    contours, hierarchy = cv2.findContours(mask_angle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest area for noise condition
    areas = [cv2.contourArea(c) for c in contours]
    
    if not areas:
        hidden = True
        return pose_hidden, hidden, mask_angle
        
    max_cont = max(areas)
    
    # Hidden camera condition
    if len(contours) == 0: 
        hidden = True
        return pose_hidden, hidden, mask_angle
    
    
    triangle_found=0
    square_found=0
   
    # For each contour found, approximate it as a polygon and extract its corners if it is a rectangle/triangle
    for cont in contours:
        
        epsilon = POLY_FACTOR_ROB * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        
        if(len(approx)>2 and cv2.contourArea(approx) >= max_cont/6): # Condition to get rid of detected noise
            if(len(approx)==3):
                corners[0] = approx
                triangle_found=1
                
            if(len(approx)==4):
                corners[1] = approx
                square_found=1
                
    # Hidden camera condition
    if square_found == 0 or triangle_found == 0:
        hidden = True
        return pose_hidden, hidden, mask_angle
    
    # From corners, get centers of rectangle and triangle
    for i in range(0, len(corners)):
        if(len(corners[i]) == 4):
            center_rect = centroid(corners[i])
            centers.append(center_rect)
            (x,y) = center_rect # Center of the robot is the center of the red rectangle
        else:
            centers.append(centroid(corners[i]))

    diff = [centers[0][0]-centers[1][0], centers[0][1]-centers[1][1]] # Distance between the two shapes 
    angle = math.degrees(np.arctan2(diff[1], diff[0])%(2*np.pi))
    
    pose = np.array([x*px_factor, y*px_factor, angle])
    
    return pose, hidden, mask_angle

def get_image(cap):
    # Iterates through the video capture buffer until we extract the most recent frame captured by the camera
    
    # param cap : header to access camera
    
    # return frame : most recent frame captured by the camera
    
    while True:
        # Compute the time necessary to read one frame 
        previous = time.time() 
        ret, frame = cap.read()
        actual = time.time()
        
        if not ret:
            raise ValueError("Can not read frame")
        
        # If the time taken to read one frame is higher than 20ms, the frame was just captured and not taken from the buffer
        if actual-previous>0.02:
            break

    x = X_CROP_LEFT
    y = Y_CROP_TOP
    w = X_CROP_RIGHT
    h = Y_CROP_BOT
    
    # Crop frame to fit the map 
    frame = frame[y:y+h, x:x+w]
    return frame

def display_obstacle(image, start, goal, obstacle):
    # Displays the expanded form of the obstacles extracted from the obstacle() function  
    # Also displays the starting position of the Thymio and the goals centers
    
    # param image : current frame
    # param start : coordinates of the starting position of the Thymio
    # param goal : coordinates of the centers of the goals 
    # param obstacle : coordinates of the expanded corners of the obstacles 
    
    shape = np.zeros_like(image, np.uint8)
    
    cv2.circle(shape, start,radius=0, color=(0,255,0), thickness=15)
    for current_goal in goal:
        cv2.circle(shape, current_goal,radius=0, color=(0,255,255), thickness=15)
    cv2.fillPoly(shape, obstacle, color=(0,0,255))
    
    alpha = 0.5
    mask = shape.astype(bool)
    image[mask] = cv2.addWeighted(image, alpha, shape, 1 - alpha, 0)[mask]
    return

def display_pos(image, pos, px_to_mm, hidden_cam, label):
    # Displays either: the position of the robot as computed by the vision (label = 1)
    #                  the position of the robot as estimated by the Kalman filter (label = 0)
    #                  the position of the current goal towards which the robot is going (label = 2)
    
    # param image : current frame
    # param pos : coordinates of the position we want to display
    # param px_to_mm : pixel to millimeter factor
    # param hidden_cam : boolean for hidden camera condition
    # param label : determines what position we are displaying 
    
    if hidden_cam: # if camera is hidden, display nothing
        return
    posa = np.array([pos[0],pos[1]])
    posa = np.int0(posa/px_to_mm).tolist()
    if label ==1:
        cv2.circle(image, posa, radius=0, color=(0,255,0), thickness=15)
    if label ==2:
        cv2.circle(image, posa, radius=0, color=(0,0,0), thickness=15)
    else:
        cv2.circle(image, posa, radius=0, color=(255,0,0), thickness=15)
        
    return
        

