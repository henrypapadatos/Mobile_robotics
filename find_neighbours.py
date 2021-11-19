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
            if (self.get_position()[0] == current_point.get_position()[0]) and (self.get_position()[1] == current_point.get_position()[1]):
                continue
            
            for obstacle_line in obstacle_lines:
                if self.intersect(current_point, obstacle_line):
                    break                
            self.neighbours.append(current_point)
        return
    
    def get_position(self):
        return np.array([self.x,self.y])
    
    def intersect(self, current_point, osbtacle_line): 
 
        p1 = self
        q1 = current_point
        p2 = obstacle_line.A
        q2 = obstacle_line.B
        
        # Find the 4 orientations required for
        # the general and special cases
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
     
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
     
        # Special Cases
     
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True
     
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True
     
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True
     
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True
     
        # If none of the cases
        return False
    
    def onSegment(p, q, r):
        # Given three collinear points p, q, r, the function checks if
        # point q lies on line segment 'pr'
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
               (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
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
         
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if (val > 0):
             
            # Clockwise orientation
            return 1
        elif (val < 0):
             
            # Counterclockwise orientation
            return 2
        else:
             
            # Collinear orientation
            return 0
    
class obstacle_line:
    def __init__(self, A, B):
        self.A = A
        self.B = B
    
    def get_obstacle_line(self):
        return np.array([self.A.get_position(), self.B.get_position()])
    
def set_up_lists(start, goal, obstacle):   
    point_list = []
    obstacle_line_list = []
    
    point_list.append(point(start))
    for current_goal in goal:
        point_list.append(point(current_goal))
        
    for current_obstacle in obstacle:
        A = point(current_obstacle[-1])
        point_list.append(A)
        obstacle_line_list.append(obstacle_line(A, point(current_obstacle[0])))
        for i in range (len(current_obstacle)-1):
            A = point(current_obstacle[i])
            point_list.append(A)
            obstacle_line_list.append(obstacle_line(A, point(current_obstacle[i+1])))
        
    return point_list, obstacle_line_list


    

#test values
start = np.array([5,5])

goal = []
goal.append(np.array([450,80]))
goal.append(np.array([300,300]))

obstacle = []
obstacle.append(np.array( [[10,50], [400,50], [90,200], [50,500]]))
obstacle.append(np.array( [[250,350], [400,500], [500,500]])) 

point_list, obstacle_line_list = set_up_lists(start, goal, obstacle)

for point in point_list:
    point.find_neighbours(obstacle_line_list,point_list)

# Create a black image
image = np.zeros((512,512,3), np.uint8)

#draw obstacle
for current_obstacle in  obstacle_line_list:
    image = cv2.line(image, current_obstacle.A.get_position(), current_obstacle.B.get_position(), color=(0,0,255), thickness = 3)

cv2.imshow("Black Rectangle (Color)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


