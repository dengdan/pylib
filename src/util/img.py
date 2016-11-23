#coding=utf-8
'''
@author: dengdan
'''
import cv2
import numpy as np
import logging
import math

import util

IMREAD_GRAY = 0
IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1



COLOR_WHITE =(255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)

COLOR_RGB_RED = (255, 0, 0)
COLOR_BGR_RED = (0, 0, 255)

COLOR_BGR_YELLOW = (0, 255, 255)
COLOR_BGR_RED = (0, 0, 255)

def imshow(winname, img, mode = IMREAD_UNCHANGED, block = True, position = None):
    if not isinstance(img, np.ndarray):
        img = imread(path = img, mode = mode)
        
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    if position is not None:
        cv2.moveWindow(winname, position[0], position[1])
        
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def move_win(winname, position = (0, 0)):
    """
    move pyplot window
    """
    cv2.moveWindow(winname, position[0], position[1])

def cvt_white(img, color = COLOR_BGR_YELLOW):
    h, w, _ = img.shape
    for y in range(1, h):
            for x in range(1, w):
                if is_white(img[y, x]):
                    img[y, x] = color
    return img

def change_color(img, target, color = COLOR_GREEN):
    """convert pixels of color 'target' to color 'color'"""
    h, w, _ = img.shape
    for y in range(1, h):
            for x in range(1, w):
                if eq_color(img[y, x], target):
                    img[y, x] = color
    return img


def eq_color(target, color):
    for i, c in enumerate(color):
        if target[i] != color[i]:
            return False
    return True
    
def is_white(color):
    for c in color:
        if c < 255:
            return False
    return True
    
def black(shape):
    if len(np.shape(shape)) >= 2:
        shape = get_shape(shape)
    return np.zeros(shape, np.uint8)
    
def white(shape):
    if len(np.shape(shape)) >= 2:
        shape = get_shape(shape)
    return np.ones(shape, np.uint8) * 255
    
def imread(path, rgb = False, mode = cv2.IMREAD_COLOR):
    path = util.io.get_absolute_path(path)
    img = cv2.imread(path, mode)
    if img is None:
        raise IOError('File not found:%s'%(path))
    if rgb:
        img = bgr_to_rgb(img)
    return img
    
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bgr_to_rgb = bgr2rgb

def ds_size(image_size, kernel_size, stride):
    """calculate the size of downsampling result"""
    image_x, image_y = image_size
    kernel_x, kernel_y = kernel_size
    stride_x, stride_y = stride
    
    def f(iw, kw, sw):
        return int(np.floor((iw - kw) / sw) + 1)
    
    output_size = (f(image_x, kernel_x, stride_x), f(image_y, kernel_y, stride_y))
    return output_size


    
def get_roi(img, p1, p2):
    """
    extract region of interest from an image.
    p1, p2: two tuples standing for two opposite corners of the rectangle bounding the roi. 
    Their order is arbitrary.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    x_min = min([x1, x2])
    y_min = min([y1, y2])
    x_max = max([x1, x2]) + 1
    y_max = max([y1, y2]) + 1
    
    return img[y_min: y_max, x_min: x_max]
    

def rotate_about_center(src, angle, scale=1.):
    """https://www.oschina.net/translate/opencv-rotation"""
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)



def rectangle(img, left_up, right_bottom, color, border_width = 1):
    cv2.rectangle(img, left_up, right_bottom, color, border_width)

def rect_perimeter(left_up, right_bottom):
    """
    calculate the perimeter of the rectangle described by its left-up and right-bottom point.
    """
    return sum(np.asarray(right_bottom) -  np.asarray(left_up)) * 2

def apply_mask(img, mask):
    """
    the img will be masked in place. 
    """
    c = np.shape(img)[-1]
    for i in range(c):
        img[:, :, i] = img[:, :, i] * mask 
    return img
    
def get_shape(img):
    """
    return the height and width of an image
    """
    return np.shape(img)[0:2]
    
def get_value(img, x, y = None):
    if y == None:
        y = x[1]
        x = x[0]
    
    return img[y][x]        
    
def set_value(img, xy, val):
    x, y = xy
    img[y][x] = val


def filter2D(img, kernel):
    dst = cv2.filter2D(img, -1, kernel)
    return dst

def average_blur(img, shape = (5, 5)):
    return cv2.blur(img, shape)

def gaussian_blur(img, shape = (5, 5), sigma = 0):
    # sigma --> sigmaX, sigmaY
    blur = cv2.GaussianBlur(img,shape, sigma)
    return blur

def bilateral_blur(img, d = 9, sigmaColor = 75, sigmaSpace = 75):
    dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    return dst

BLUR_AVERAGE =  'average'
BLUR_GAUSSIAN = 'gaussian'
BLUR_BILATERAL = 'bilateral'


_blur_dict = {
              BLUR_AVERAGE: average_blur,
              BLUR_GAUSSIAN: gaussian_blur,
              BLUR_BILATERAL: bilateral_blur
}
def blur(img, blur_type):
    fn = _blur_dict[blur_type]
    return fn(img)
