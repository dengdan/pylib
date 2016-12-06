#coding=utf-8
'''
@author: dengdan
'''
import cv2
import numpy as np
import logging
import math
import event
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
    
def imread(path, rgb = False, mode = cv2.IMREAD_COLOR):
    path = util.io.get_absolute_path(path)
    img = cv2.imread(path, mode)
    if img is None:
        raise IOError('File not found:%s'%(path))
    if rgb:
        img = bgr2rgb(img)
    return img

def imshow(winname, img, mode = IMREAD_UNCHANGED, block = True, position = None, maximized = False):
    if not isinstance(img, np.ndarray):
        img = imread(path = img, mode = mode)
    
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    if position is not None:
#         cv2.moveWindow(winname, position[0], position[1])
        move_win(winname, position)
    
    if maximized:
        maximize_win(winname)  
        
    if block:
#         cv2.waitKey(0)
        event.wait_key(" ")
        cv2.destroyAllWindows()


def imwrite(path, img):
    path = util.io.get_absolute_path(path)
    util.io.make_parent_dir(path)
    cv2.imwrite(path, img)

def move_win(winname, position = (0, 0)):
    """
    move pyplot window
    """
    cv2.moveWindow(winname, position[0], position[1])

def maximize_win(winname):
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, True);

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
    
def white(shape, value = 255):
    if len(np.shape(shape)) >= 2:
        shape = get_shape(shape)
    return np.ones(shape, np.uint8) * value
    
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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
    
def rectangle(img, left_up, right_bottom, color, border_width = 1):
    cv2.rectangle(img, left_up, right_bottom, color, border_width)


def circle(img, center, r, color, border_width = 1):
    cv2.circle(img, center, r, color, border_width)

def render_points(img, points, color):
    for p in points:
        x, y = p
        img[y][x] = color

    
def draw_contours(img, contours, idx = -1, color = 1, border_width = 1):
    cv2.drawContours(img, contours, idx, color, border_width)
    return img

def get_contour_rect_box(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x, y, w, h

def get_contour_region_in_rect(img, contour):
    x, y, w, h = get_contour_rect_box(contour)
    lu, rb = (x, y), (x + w, y + h)
    return get_roi(img, lu, rb)

def get_contour_min_area_box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    return box

def get_contour_region_in_min_area_rect(img, cnt):
    # find the min area rect of contour
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    box = cv2.cv.BoxPoints(rect)
    box_cnt = points_to_contour(box)
    
    # find the rectangle containing box_cnt, and set it as ROI
    outer_rect = get_contour_rect_box(box_cnt)
    x, y, w, h = outer_rect
    img = get_roi(img, (x, y), (x + w,  y + h))
    box = [(ox - x, oy - y) for (ox, oy) in box]
    
    # rotate ROI and corner points
    rows, cols = get_shape(img)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, scale = 1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    bar_xy = np.hstack((box, np.ones((4, 1))))
    new_corners = np.dot(M, np.transpose(bar_xy))
    new_corners = util.dtype.int(np.transpose(new_corners))
#     cnt = points_to_contour(new_corners)
    
    xs = new_corners[:, 0]
    ys = new_corners[:, 1]
    lu = (min(xs), min(ys))
    rb = (max(xs), max(ys))
    return get_roi(dst, lu, rb)


def contour_to_points(contour):
    return np.asarray([c[0] for c in contour])


def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])

def get_contour_region_iou(I, cnt1, cnt2):
    """
    calculate the iou of two contours
    """
    mask1 = util.img.black(I)
    draw_contours(mask1, [cnt1], color = 1, border_width = -1)
    
    mask2 = util.img.black(I)
    draw_contours(mask2, [cnt2], color = 1, border_width = -1)
    
    union_mask = ((mask1 + mask2) >=1) * 1
    intersect_mask = (mask1 * mask2 >= 1) * 1
    
    return np.sum(intersect_mask) * 1.0 / np.sum(union_mask)

    
def fill_bbox(img, box, color = 1):
    """
    filling a bounding box with color.
    box: a list of 4 points, in clockwise order, as the four vertice of a bounding box
    """
    util.test.assert_equal(np.shape(box), (4, 2))
    cnt = to_contours(box)
    draw_contours(img, cnt, color = color, border_width = -1)
    
def get_rect_points(left_up, right_bottom):
    """
    given the left up and right bottom points of a rectangle, return its four points
    """
    right_bottom, left_up = np.asarray(right_bottom), np.asarray(left_up)
    w, h = right_bottom - left_up
    x, y = left_up
    points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return points
    
def rect_perimeter(left_up, right_bottom):
    """
    calculate the perimeter of the rectangle described by its left-up and right-bottom point.
    """
    return sum(np.asarray(right_bottom) -  np.asarray(left_up)) * 2

def rect_area(left_up, right_bottom):
    wh = np.asarray(right_bottom) - np.asarray(left_up) + 1
    return np.prod(wh)
    
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

def get_wh(img):
    return np.shape(img)[0:2][::-1]

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
    
def put_text(img, text, pos, scale = 1, color = COLOR_WHITE, thickness = 1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img = img, text = text, org = tuple(pos), fontFace = font,  fontScale = scale,  color = color, thickness = thickness)

def resize(img, f = None, fx = None, fy = None, size = None):
    h, w = get_shape(img)
    if fx != None and fy != None:
        return cv2.resize(img, None, fx = fx, fy = fy)
        
    if size != None:
        size = tuple(util.dtype.int(size))
        return cv2.resize(img, size)
    
    return cv2.resize(img, None, fx = f, fy = f)

def translate(img, delta_x, delta_y, size = None):
    M = np.float32([[1,0, delta_x],[0,1, delta_y]])
    if size == None:
        size = get_wh(img)
    
    dst = cv2.warpAffine(img,M, size)
    return dst


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

