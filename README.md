# Portfolio
These programs are intended for use after using ViT (Vision Transformers) Pose, a top-down pose estimation system.
ViTPose is a computer vision model using a transformer architecture designed to estimate an individual's pose in complex and cluttered environments.
It was released in 2021 and outperforms state-of-the-art models on a variety of benchmark datasets.

These programs implement head movement analysis with Tracking each individual from there.

# Tracking
The ViTPose used as a posture estimation system takes video as input, but actually detects keypoints for each image, so there is no connection between keypoints between images.
Therefore, there is no connection between the keypoints of the images. This creates the problem of not being able to track the movements of individuals.
To solve this problem, we use as input video that has been subjected to a process called cropping, which cuts out all but the individual to be tracked.
However, it is not always possible to crop the motion range of the individual to be tracked well, and other students may be included.
In addition, there are cases where a single person is detected more than once.
Therefore, considering these problems, we created a program to track each of them.
The program consists of two components.

First, it is consistent within a frame.(1.CleanIndividuals.py)

Next, the frames are matched to each other.(2.TrackingID.py)

However, it does not completely prevent ID switching, and long-time tracking is a future challenge.

## 1.CleanIndividuals.py
In order to keep the number of people detected per frame constant, the median points of the nose, right shoulder, and left shoulder of each detected person are used.
The over-detected persons are integrated as the same person using the k-means method. Integration is performed by taking the average of each key point.
In this case, persons below a certain accuracy are deleted.

## 2.TrackingID.py
The distance between the median of the nose, right shoulder, and left shoulder of each detected person is taken between frames, and the person with the smallest distance is considered to be the same person.
If the accuracy is lower than a certain level, linear completion is performed later.

# analysis
The time-series data of posture for each person obtained by the above process is analyzed.
This time, we focus on head movements.(3.Head_movement_analysis.py)

## 3.Head_movement_analysis.py
Preprocessing
1.Linear completion (missing value completion)
2.Low-pass filter (5Hz or lower)
3.moving average (window width 15, FPS=30)
4.normalize the transition waveforms of x,y coordinates by the width of x-axis and y-axis of shoulder width each

Trajectory (total trajectory length)
The trajectory is the time series data plotted on the x and y coordinates of the camera.
The total trajectory length is the calculated length of the trajectory.

Displacement per frame
Displacement per frame of video

Power spectrum
Distribution of rough and fine movements

Sample entropy

Root Mean Square (Root Mean Square, RMS)
Evaluate the variability of the movement.
Scattering of movement magnitude

## 4.DTW.py
Dynamic Time Warping (DTW)
Synchronization of movements (similarity)
Find the distance between each point of two time series by summing
Sum of paths that have the shortest distance between the two time series



## 5.opticalflow.py
Dense optical flow summed over video time and normalized by maximum RGB value
This program is not related to ViTPose, only to evaluate objects in the image.
