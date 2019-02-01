from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import util

path = util.argv[1]
src = util.img.imread(path)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

min_hessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold = min_hessian)
keypoints = detector.detect(src_gray)
img = util.img.black(src)
img = cv.drawKeypoints(src, keypoints, img)
cv.imshow("SURF", img)
cv.waitKey()
