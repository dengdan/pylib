from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import pdb
import util

path1 = util.argv[1]
path2 = util.argv[2]
src1 = util.img.imread(path1)
src2 = util.img.imread(path2)
src1_gray = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
src2_gray = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)

min_hessian = 10000
detector = cv.xfeatures2d_SURF.create(hessianThreshold = min_hessian)
keypoints1, des1 = detector.detectAndCompute(src1_gray, None)
keypoints2, des2 = detector.detectAndCompute(src2_gray, None)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
matches = matcher.match(des1, des2)

for m in matches:
    img_matches = np.empty((src1.shape[0], src2.shape[1] * 2, 3), dtype = np.uint8)
    cv.drawMatches(src1_gray, keypoints1, src2_gray, keypoints2, [m], img_matches)
    cv.imshow("SURF", img_matches)
    cv.waitKey()
