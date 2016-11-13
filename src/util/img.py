#coding=utf-8
'''
Created on 2016年9月29日

@author: dengdan
'''
import cv2
import matplotlib.patches as patches
import numpy as np
import logging

import util.io

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

def imshow(winname, img, mode = IMREAD_UNCHANGED, block = False, position = None):
    if not isinstance(img, np.ndarray):
        img = imread(path = img, mode = mode)
        
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    if position is not None:
        cv2.moveWindow(winname, position[0], position[1])
        
    if block:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
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
    
def fullscreen(plt):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
def black(shape):
    return np.zeros(shape, np.uint8)
    
    
def imread(path, rgb = False, mode = IMREAD_UNCHANGED):
    path = util.io.get_absolute_path(path)
    img = cv2.imread(path, IMREAD_UNCHANGED)
    if img is None:
        raise IOError('File not found:%s'%(path))
    if rgb:
        img = bgr_to_rgb(img)
    return img
    
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def ds_size(image_size, kernel_size, stride):
    """calculate the size of downsampling result"""
    image_x, image_y = image_size
    kernel_x, kernel_y = kernel_size
    stride_x, stride_y = stride
    
    def f(iw, kw, sw):
        return int(np.floor((iw - kw) / sw) + 1)
    
    output_size = (f(image_x, kernel_x, stride_x), f(image_y, kernel_y, stride_y))
    return output_size

def move_win(winname, position = (0, 0)):
    cv2.moveWindow(winname, position[0], position[1])
    
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
    
def rectangle(xy, width, height, color = 'red', linewidth = 1, fill = False, alpha = None,picker = None, axis = None, contains = None):
    rect = patches.Rectangle(
        xy = xy,
        width = width,
        height = height,
        contains = contains,
        alpha = alpha,
        color = color,
        fill = fill,
        picker = picker,
        linewidth = linewidth
    )
    if axis is not None:
        axis.add_patch(rect)
    return rect
    
rect = rectangle

def line(xy_start, xy_end, color = 'red', linewidth = 1, alpha = None, axis = None):
    #logging.debug('drawing a straight line from %s to %s'%(str(xy_start), str(xy_end)))
    from matplotlib.lines import Line2D 
    num = 100
    xdata = np.linspace(xy_start[0], xy_end[0], num = num)
    ydata = np.linspace(xy_start[1], xy_end[1], num = num)
    line = Line2D(
        alpha = alpha,
        color = color,
        linewidth = linewidth,
        xdata = xdata,
        ydata = ydata
    )
    if axis is not None:
        axis.add_line(line)
    return line

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

