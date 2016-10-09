#coding=utf-8
'''
Created on 2016年9月29日

@author: dengdan
'''
import cv2
import numpy as np
IMREAD_GRAY = 0
IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BGR_YELLOW = (0, 255, 255)
def imshow(winname, img, mode = IMREAD_UNCHANGED, block = True):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img, mode)
        
    cv2.imshow(winname, img)
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def black(shape):
    return np.zeros(shape, np.uint8)
