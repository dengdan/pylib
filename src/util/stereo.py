from __future__ import division
import cv2
import numpy as np
import math
import json
from collections import OrderedDict
import pdb
import util

def _cvt_point(p):
    if np.ndim(p) == 1:
        p = [p]
    if np.ndim(p) == 2:
        return np.asarray([[[pt[0], pt[1]] for pt in p]], dtype = np.float32)
    assert np.ndim(p) == 3
    return p

def fov(fx, w):
    tan = w / fx * .5
    return np.arctan(tan) * 2 / np.pi * 180.0

def min_mono_visible_distance(vertical_fov, install_height):    
    return install_height * np.tan(vertical_fov / 180 * np.pi)

def min_bino_visible_distance(fx, w, B):
    return fx / w * B

class JsonObject(object):
    _fields = []
    def __init__(self, **data):
        if type(self._fields) == str:
            self._fields = self._fields.split(",")
        for f in self._fields:
            self.__setattr__(f, None)
        if data is not None:
            JsonObject.load(self, data)
            
    def load(self, data):
        for f in self._fields:
            v = None
            if f in data:
                v = data[f]
            self.__setattr__(f, v)
    
    def as_dict(self, excluded = None, **kwargs):
        data = OrderedDict()
        for f in self._fields:
            if not excluded or f not in excluded:
                data[f] = self.__getattribute__(f)
        return data
    
    def dump(self, path, **kwargs):
        s = self.dumps(**kwargs)
        parent_path = os.path.dirname(path)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        with open(path, "w") as f:
            f.write(s)
            
    def dumps(self, excluded = None, *args, **kwargs):
        def default(o):
            if isinstance(o, JsonObject):
                return o.as_dict(excluded = excluded, **kwargs)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return json.JSONEncoder.default(self, o)

        if "indent" not in kwargs:
            kwargs["indent"] = True

        return json.dumps(self, default = default,**kwargs)
    
    def __repr__(self, *args, **kwargs):
        return self.dumps(*args, **kwargs)
    
    def __str__(self, *args, **kwargs):
        return self.dumps(*args, **kwargs)

class CameraModel(JsonObject):
    _fields = "fx,fy,cx,cy,k1,k2,p1,p2,k3,size,P,R,alpha"
    def __init__(self, **kwargs):
        if "k3" not in kwargs:
            kwargs["k3"] = 0
        
        JsonObject.__init__(self, **kwargs)
        self.intrinsics = np.eye(3, dtype = np.float64)
        self.distortion = np.zeros((5, 1), np.float64) # rational polynomial
        self.intrinsics[0, 0] = self.fx
        self.intrinsics[1, 1] = self.fy
        self.intrinsics[0, 2] = self.cx
        self.intrinsics[1, 2] = self.cy
        self.distortion[0][0] = self.k1
        self.distortion[1][0] = self.k2
        self.distortion[2][0] = self.p1
        self.distortion[3][0] = self.p2
        self.size = tuple(self.size)
        self.w, self.h = self.size
        if self.P is not None:
            self.P = np.asarray(self.P)
        else:
            self.P = np.zeros((3, 4))
        self.R = np.asarray(self.R)
        self.set_alpha(self.alpha)
        
    def set_alpha(self, a):
        """
        Set the alpha value for the calibrated camera solution.  The alpha
        value is a zoom, and ranges from 0 (zoomed in, all pixels in
        calibrated image are valid) to 1 (zoomed out, all pixels in
        original image are in calibrated image).
        """
        self.alpha = a
        if a is not None:
            ncm, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.distortion, self.size, a)
        else:
            ncm = self.intrinsics
        for j in range(3):
            for i in range(3):
                self.P[j,i] = ncm[j, i]
        self.fx = self.P[0, 0]
        self.fy = self.P[1, 1]
        self.cx = self.P[0, 2]
        self.cy = self.P[1, 2]
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.intrinsics, self.distortion, 
                               self.R, ncm, self.size, cv2.CV_32FC1)
        
    def camera2normizedimage(self, x, y, z, homo = False):
        coord = [[x / z], [y / z]]
        if homo:
            coord.append([1])
        return np.asanyarray(coord)
    
    def pixel2normalizedimage(self, u, v, homo = False):
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        coord = [[x], [y]]
        if homo:
            coord.append([1])
        return np.asarray(coord)
    
    def undistort_point(self, x, y):
        src = np.asarray([[[x, y]]], dtype = np.float32)
        points = self.undistort_points(src)
        return tuple(points[0][0].tolist())
    
    def undistort_points(self, points):
        """
        points: N source pixel points (u,v) as an Nx2 matrix
        """
        points = np.asarray(points, dtype = np.float32)
        if len(points.shape) == 2:
            assert points.shape[1] == 2
            points = np.expand_dims(points, axis = 1)
        assert points.shape[1] == 1
        return cv2.undistortPoints(points, self.intrinsics, self.distortion, R = self.R, P = self.P)

    def remap(self, src):
        """
        :param src: source image
        :type src: :class:`cvMat`

        Apply the post-calibration undistortion to the source image
        """
        return cv2.remap(src, self.mapx, self.mapy, cv2.INTER_LINEAR)
    
    def fov(self):
        return fov(self.fx, self.w), fov(self.fy, self.h)
    
    def visible_range(self, install_height):
        xfov, yfov = self.fov()
        return min_visible_distance(yfov, install_height)

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
                opts_loc[j, :] = opts_loc[j, :] * self.square
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
        if corners is None:
            ok, corners = self.find_corners(img, refine = True) 
        else:
            ok = True
        if not ok:
            return img
        cv2.drawChessboardCorners(img, (self.n_cols, self.n_rows), corners, True)
        return img

class BinocularModel(JsonObject):
    _fields = "left_camera_model,right_camera_model,R,T"
    def __init__(self, **data):
        JsonObject.__init__(self, **data)
        if self.R is None:
            self.R = np.eye(3, dtype = np.float64)
        if self.T is None:
            self.T = np.zeros((3, 1))
        self.Q = np.zeros((4, 4))
        self.R = np.asarray(self.R)
        self.T = np.asarray(self.T)
        self.B = abs(self.T[0])
        if not isinstance(self.left_camera_model, CameraModel):
            self.left_camera_model = CameraModel(**self.left_camera_model)
        if not isinstance(self.right_camera_model, CameraModel):
            self.right_camera_model = CameraModel(**self.right_camera_model)
        self.set_alpha(0)
        
    def set_alpha(self, a = -1):
        dl = self.left_camera_model.distortion
        dr = self.right_camera_model.distortion
        cv2.stereoRectify(self.left_camera_model.intrinsics,
                         self.left_camera_model.distortion,
                         self.right_camera_model.intrinsics,
                         self.right_camera_model.distortion,
                         self.left_camera_model.size,
                         self.R,
                         self.T,
                         self.left_camera_model.R, self.right_camera_model.R, 
                         self.left_camera_model.P, self.right_camera_model.P,
                         self.Q, alpha = a)
        cv2.initUndistortRectifyMap(self.left_camera_model.intrinsics, dl,
                                    self.left_camera_model.R, 
                                    self.left_camera_model.P, 
                                    self.left_camera_model.size, cv2.CV_32FC1,
                                    self.left_camera_model.mapx, 
                                    self.left_camera_model.mapy)
        
        cv2.initUndistortRectifyMap(self.right_camera_model.intrinsics, dr, 
                                    self.right_camera_model.R, 
                                    self.right_camera_model.P, 
                                    self.right_camera_model.size, cv2.CV_32FC1,
                                    self.right_camera_model.mapx, 
                                    self.right_camera_model.mapy)
    
    def dispairity(self, lpoints, rpoints, distorted = False):
        lpoints = _cvt_point(lpoints)
        rpoints = _cvt_point(rpoints)
        if distorted:
            lpoints = self.left_camera_model.undistort_points(lpoints)
            rpoints = self.right_camera_model.undistort_points(rpoints)
        return lpoints[..., 0] - rpoints[..., 0]
    
    def deproject(self, lpoints, rpoints):
        lpoints = self.left_camera_model.undistort_points(lpoints)
        rpoints = self.right_camera_model.undistort_points(rpoints)
        ds = self.dispairity(lpoints, rpoints)
        homos = np.concatenate([lpoints[:, 0, :], ds, np.ones_like(ds)], axis = 1)
        Homos = np.dot(self.Q, np.transpose(homos, [1, 0])).transpose([1, 0])
        Homos = Homos / Homos[:, 3:]
        return Homos[:, :3]
    
    def get_depth(self, distored_lp, distored_rp):
        coords = self.deproject(_cvt_point(distored_lp), _cvt_point(distored_rp))
        return coords[0][2]
    
    def min_visible_distance(self):
        """
        the min visible distance of binocular system
        """
        return min_bino_visible_distance(self.left_camera_model.fx, self.left_camera_model.w, self.B)
    
