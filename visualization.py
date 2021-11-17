# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 19:59:15 2021

@author: henry papadatos
"""

import cv2
import numpy as np

# Create a black image
image = np.zeros((512,512,3), np.uint8)

#define start position
start = np.array([5,5])
cv2.circle(image, start,radius=0, color=(0,255,0), thickness=5) 

#define goals positions
goal = []
goal.append(np.array([450,80]))
goal.append(np.array([300,300]))
for current_goal in goal:
    cv2.circle(image, current_goal,radius=0, color=(0,255,255), thickness=5) 

#define a list with all the obstacles
obstacle = []
obstacle.append(np.array( [[10,50], [400,50], [90,200], [50,500]]))
obstacle.append(np.array( [[250,350], [400,500], [500,500]]))

# #draw all obsatcle
# cv2.polylines(image, obstacle, True, color=(0,0,255), thickness=-1)
cv2.fillPoly(image, obstacle, color=(0,0,255))

cv2.imshow("Black Rectangle (Color)", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

