from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import util

source_window = "Image"
max_tracker_bar = 100
rng.seed(12345)

path = util.argv[1]
src = util.img.imread(path)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

def good_features_to_track(val):
    max_corners = max(val, 1)
    
    # params for Shi-Tomasi algorithm
    quality_level = .01
    min_distance = 10
    block_size = 3
    gradient_size = 3
    use_harris_detector = False
    k = 0.04
    
    copy = np.copy(src) 
    corners = cv.goodFeaturesToTrack(src_gray, max_corners, quality_level, min_distance, 
         None, blockSize = block_size, gradientSize = gradient_size, useHarrisDetector = use_harris_detector, k = k)
    print("Number of corners detected", corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        util.img.circle(copy, center = (corners[i, 0, 0], corners[i, 0, 1]), 
                        r = radius, color = util.img.random_color_3())
    
    cv.imshow(source_window, copy)
    
cv.namedWindow(source_window, cv.WINDOW_NORMAL)
max_corners = 23
cv.createTrackbar("Threshold: ", source_window, max_corners, max_tracker_bar, good_features_to_track)
good_features_to_track(max_corners)
cv.waitKey()