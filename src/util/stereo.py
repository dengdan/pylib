from __future__ import division
import cv2
import numpy as np
import math

class CameraModel(object):
    _param_names = "fx,fy,cx,cy,k1,k2,p1,p2,k3".split(",")
    def __init__(self, kwargs):
        if "k3" not in kwargs:
            kwargs["k3"] = 0
            
        for name in self._param_names:
            self.__setattr__(name, kwargs[name])
        self.intrinsics = np.eye(3, dtype = np.float64)
        self.distortion = np.zeros((8, 1), np.float64) # rational polynomial
        self.intrinsics[0, 0] = self.fx
        self.intrinsics[1, 1] = self.fy
        self.intrinsics[0, 2] = self.cx
        self.intrinsics[1, 2] = self.cy
        self.distortion[0][0] = self.k1
        self.distortion[1][0] = self.k2
        self.distortion[2][0] = self.p1
        self.distortion[3][0] = self.p2

class CalibrationBoard(object):
    class Patterns:
        Chessboard, Circles, ACircles = "chessboard", "circles", "acircles"
    def __init__(self, n_rows, n_cols, square = 108, pattern = None):
        self.pattern = pattern
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.square = square
        if self.pattern is None:
            self.pattern = self.__class__.Patterns.Chessboard

    def get_object_points(self, use_board_size = False, expand_dim = True):
        opts = []
        num_pts = self.n_cols * self.n_rows
        opts_loc = np.zeros((num_pts, 3), np.float32)
        for j in range(num_pts):
            opts_loc[j, 0] = j // self.n_cols
            if self.pattern == CalibrationBoard.Patterns.ACircles:
                opts_loc[j, 1] = 2 * (j % self.n_cols) + (opts_loc[j, 0, 0] % 2)
            else:
                opts_loc[j, 1] = (j % self.n_cols)
            opts_loc[j, 2] = 0
            if use_board_size:
                opts_loc[j, :] = opts_loc[j, :] * self.dim
        if expand_dim:
            opts_loc = np.expand_dims(opts_loc, axis = 1)
        return opts_loc
        
    def find_corners(self, img, refine = True):
        if len(np.shape(img)) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.pattern == CalibrationBoard.Patterns.Chessboard:
            ok, corners = cv2.findChessboardCorners(img, (self.n_cols, self.n_rows), None)
            if not ok:
                return ok, corners
        def _pdist(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
        min_distance = np.infty
        num_points = self.n_cols * self.n_rows
        for idx in range(num_points - 1):
            right = idx + 1
            down = idx + self.n_cols
            if right // self.n_cols == idx // self.n_cols:
                min_distance = min([min_distance, _pdist(corners[idx, 0], corners[right, 0])])
            if down < num_points:
                min_distance = min([min_distance, _pdist(corners[idx, 0], corners[down, 0])])
        
        radius = int(math.ceil(min_distance * 0.5))

        if ok and refine:
            cv2.cornerSubPix(img, corners, (radius,radius), (-1,-1),
                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 ))
        return ok, corners
    
    def draw_corners(self, img, corners = None):
        ok = False
        if not corners:
            ok, corners = self.find_corners(img, refine = True) 
        if not ok:
            return img
        
        cv2.drawChessboardCorners(img, (self.n_cols, self.n_rows), corners, True)
        return img

