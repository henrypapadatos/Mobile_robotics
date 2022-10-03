# Mobile_robotics
Project of robotics that implements: vision, global navigation, local navigation and filtering.

Here is a video that demonstrates the 4 aspects of the project.
- Vision: track the position of the robot, detect the position of the obastacles
- Global navigaiton: solves the traveling salesman problem to connect all the goals (green dots) and then come back at the initial position
- Local naviagation: distance sensores are used to avoid abstacles that arise in the path
- Filtering: a kalman filter is implemented to track the position of the robot. It merges the information coming from the vision and from the odometry. Therefore, if the the camera is obstructed, the position estimation will only be based on odometry.

https://user-images.githubusercontent.com/63106608/193537901-91fc5340-23d8-4043-9816-6ba006b7d9a3.mp4

For a detailed explanation of the project, see the folder "Report.ipynb".

This project was realised in the scope of the class "Basics of mobile robotics" (MICRO-452) thaught by Mondada Francesco.
