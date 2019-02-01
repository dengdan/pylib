from __future__ import print_function
import cv2 as cv
import numpy as np
import util
import pdb


path = util.argv[1]
src = util.img.imread(path)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

source_window = "SourceImage"
corners_window = "Corners detected"
max_thresh = 255

def cornerHarris_demo(val):
    thresh = val
    img = src.copy()
    block_size = 3
    aperture_size = 3
    k = 0.04
    
    dst = cv.cornerHarris(src_gray, block_size, aperture_size, k)
    dst_norm = np.empty(dst.shape, dtype = np.float32)
    cv.normalize(dst, dst_norm, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                util.img.circle(img, (j, i), r = 5, color = util.img.COLOR_GREEN, border_width = 1)
    
    cv.imshow(source_window, img)
#     cv.imshow(corners_window, dst_norm_scaled)
    util.plt.imshow(corners_window, dst)
    util.plt.plt.show()
    
thresh = 200

cv.namedWindow(source_window)
cv.namedWindow(source_window)
cv.createTrackbar("Threshold: ", source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, src)
cornerHarris_demo(thresh)
cv.waitKey()