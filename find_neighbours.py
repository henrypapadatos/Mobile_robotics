# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:27:10 2021

@author: papad
"""

import numpy as np
import cv2

class point:
    def __init__(self, position, point_type=0):
        self.x = position[0]
        self.y = position[1]
        self.point_type = point_type
        self.neighbours = []
        
    def find_neighbours(self, obstacle_lines, other_points):
        for current_point in other_points:
            if (np.array_equal(self.get_position(),current_point.get_position())):
                continue
            
            is_neighbor = True
            for current_obstacle_line in obstacle_lines:
                if self.intersect(current_point, current_obstacle_line):
                    is_neighbor = False
                    continue   
            if is_neighbor:
                self.neighbours.append(current_point)
        return
    
    def get_position(self):
        return np.array([self.x,self.y])
    
    def get_neighbours(self):
        return self.neighbours
    
    def get_type(self):
        return self.point_type
    
    def intersect(self, current_point, current_obstacle_line): 
 
        p1 = self.get_position()
        q1 = current_point.get_position()
        p2 = current_obstacle_line.A.get_position()
        q2 = current_obstacle_line.B.get_position()
        
        if (np.array_equal(p1,p2) or np.array_equal(q1,p2) or np.array_equal(p1,q2) or np.array_equal(q1,q2)):
            return False
        
        # if (self.point_type * current_point.get_type() <= 0 or self.point_type != current_point.get_type()):
        #     if ((np.array_equal(p1,p2) or np.array_equal(q1,p2)) or (np.array_equal(p1,q2) or np.array_equal(q1,q2))):
        #         return False
        
        # #If we don't do this, the path can't be on an obstacle line 
        # if ((np.array_equal(p1,p2) and np.array_equal(q1,p2)) or (np.array_equal(p1,q2) and np.array_equal(q1,q2))):
        #     return False 
        
        # #I do this so the path can't do trough an obstacle
        # if (self.point_type == current_point.get_type() and self.point_type > 0):
        #     return True
        
        # Find the 4 orientations required for
        # the general and special cases
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
     
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
     
        # Special Cases
     
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and onSegment(p1, p2, q1)):
            return True
     
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and onSegment(p1, q2, q1)):
            return True
     
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and onSegment(p2, p1, q2)):
            return True
     
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and onSegment(p2, q1, q2)):
            return True
     
        # If none of the cases
        return False
    
    
class obstacle_line:
    def __init__(self, A, B):
        self.A = A
        self.B = B
    
    def get_obstacle_line(self):
        return np.array([self.A.get_position(), self.B.get_position()])
    
def set_up_lists(start, goal, obstacle):   
    point_list = []
    obstacle_line_list = []
    
    point_list.append(point(start, point_type = -1))
    for current_goal in goal:
        point_list.append(point(current_goal, point_type = 0))
        
    index = 1    
    for current_obstacle in obstacle:
        A = point(current_obstacle[-1], point_type=index)
        point_list.append(A)
        obstacle_line_list.append(obstacle_line(A, point(current_obstacle[0])))
        for i in range (len(current_obstacle)-1):
            A = point(current_obstacle[i], point_type=index)
            point_list.append(A)
            obstacle_line_list.append(obstacle_line(A, point(current_obstacle[i+1])))
        index += 1
        
    return point_list, obstacle_line_list

def onSegment(p, q, r):
    # Given three collinear points p, q, r, the function checks if
    # point q lies on line segment 'pr'
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False

def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
     
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
         
        # Clockwise orientation
        return 1
    elif (val < 0):
         
        # Counterclockwise orientation
        return 2
    else:
         
        # Collinear orientation
        return 0
    

#test values
start = np.array([5,5])

goal = []
goal.append(np.array([450,80]))
goal.append(np.array([300,300]))

obstacle = []
obstacle.append(np.array( [[10,50], [400,50], [90,200], [50,500]]))
obstacle.append(np.array( [[250,350], [400,500], [500,500]])) 

point_list, obstacle_line_list = set_up_lists(start, goal, obstacle)

for current_point in point_list:
    current_point.find_neighbours(obstacle_line_list,point_list)

# Create a black image
image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, start,radius=0, color=(0,255,0), thickness=5)

for current_goal in goal:
    cv2.circle(image, current_goal,radius=0, color=(0,255,255), thickness=5) 

#draw obstacle
for current_obstacle in  obstacle_line_list:
    image = cv2.line(image, current_obstacle.A.get_position(), current_obstacle.B.get_position(), color=(0,0,255), thickness = 3)
    
for current_point in point_list:
    for current_neighbour in current_point.get_neighbours():
       image = cv2.line(image, current_point.get_position(), current_neighbour.get_position(), color=(0,255,0), thickness = 1) 

cv2.imshow("Black Rectangle (Color)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


