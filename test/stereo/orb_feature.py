import util
import numpy as np
import cv2
from matplotlib import pyplot as plt

path = util.argv[1]
img = cv2.imread(path ,0)
# Initiate STAR detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORBkp, 
des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), outImage = img.copy(), flags=0)
plt.imshow(img2)
plt.show()
