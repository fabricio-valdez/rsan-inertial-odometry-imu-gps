# Inertial Odometry Complementary Filter (IMU and GPS)

This project implements an inertial odometry system for a moving vehicle by combining IMU and GPS measurements. The goal is to estimate the vehicle trajectory using inertial sensors and compare the results with GPS-based localization. The project also explores how sensor fusion techniques, such as complementary filtering, improve heading estimation and reduce drift in inertial navigation.

Course: Robotics Sensing and Navigation (RSAN)

## Tools
Hardware: VectorNav VN-100 IMU  
Hardware: Standalone GPS puck  

Data sets (previusly collected with an ROS2 package I created):  
- One ROS bag from driving in a circle (used for magnetometer calibration)  
- One ROS bag from driving around the city  

## Objectives
- Process and analyze inertial sensor data collected from a VectorNav IMU during vehicle motion.
- Apply magnetometer calibration using circular driving data to correct hard and soft iron distortions.
- Estimate vehicle heading using both magnetometer and gyroscope measurements.
- Implement a complementary filter combining low-pass filtered magnetometer data and high-pass filtered gyroscope data to obtain a more stable heading estimate.
- Estimate forward velocity using accelerometer measurements and compare it with GPS-derived velocity.
- Reconstruct the vehicle trajectory using IMU-based velocity and heading estimates.
- Compare the estimated trajectory with GPS-based localization to evaluate the performance of inertial odometry.

## Methods
- Magnetometer calibration (hard and soft iron correction)
- Gyroscope integration for yaw estimation
- Complementary filtering for heading estimation
- Velocity estimation from accelerometer data
- GPS velocity extraction
- Trajectory reconstruction from velocity and heading

## Results

Detailed plots and analysis can be found in the Results Report document.

## Figures
The report contains the following visualizations:

- Magnetometer measurements before and after calibration
- Magnetometer yaw estimation before and after calibration
- Gyroscope yaw estimation over time
- Complementary filter combining magnetometer and gyroscope heading estimates
- Forward velocity estimated from accelerometer data
- Forward velocity measured from GPS
- Comparison of GPS trajectory and IMU-based trajectory estimation

*This project comes from an assignment I completed for the Robotics Sensing and Navigation course.*
