#coding=utf-8
'''
Created on 2016年9月29日

@author: dengdan
'''
import cv2
import matplotlib.patches as patches
import numpy as np
import util.io

IMREAD_GRAY = 0
IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1



COLOR_WHITE =(255, 255, 255)
COLOR_BLACK = (0, 0, 0)

COLOR_RGB_RED = (255, 0, 0)

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
    
def black(shape):
    return np.zeros(shape, np.uint8)
    
    
def imread(path, rgb = False, mode = IMREAD_UNCHANGED):
    path = util.io.get_absolute_path(path)
    img = cv2.imread(path, IMREAD_UNCHANGED)
    if img is None:
        raise IOError('File not found:%s'%(path))
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    

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
    
def rectangle(xy, width, height, color = 'red', linewidth = 1, fill = False, alpha = None,picker = None, axes = None, contains = None):
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
    if axes is not None:
        axes.add_patch(rect)
    return rect
    
rect = rectangle

def line(xy_start, xy_end, color = 'red', linewidth = 1, alpha = None, axes = None):
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
    if axes is not None:
        axes.add_line(line)
    return line

