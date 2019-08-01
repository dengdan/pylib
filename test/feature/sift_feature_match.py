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

detector = cv.xfeatures2d_SIFT.create()
keypoints1, des1 = detector.detectAndCompute(src1_gray, None)
keypoints2, des2 = detector.detectAndCompute(src2_gray, None)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
matches = matcher.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

distances = []
for m_idx, m in enumerate(matches):
    if m_idx > 100:
        break
    left_p_idx = m.queryIdx
    right_p_idx = m.trainIdx
#     pdb.set_trace()
    left_p_y = keypoints1[left_p_idx].pt[1]
    right_p_y = keypoints2[right_p_idx].pt[1]
    dist_y = left_p_y - right_p_y
    distances.append(dist_y)
valid_count = sum([abs(d) < 5 for d in distances])
print(valid_count * 1. / len(distances))
# for m in matches:
img_matches = np.empty((src1.shape[0], src2.shape[1] * 2, 3), dtype = np.uint8)
cv.drawMatches(src1, keypoints1, src2, keypoints2, matches[:100], img_matches)
cv.imshow("SIFT", img_matches)
cv.waitKey()
