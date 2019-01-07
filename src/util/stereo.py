import cv2
import numpy as np
class CameraModel(object):
    _param_names = "fx,fy,cx,cy,k1,k2,p1,p2,k3".split(",")
    def __init__(self, **kwargs):
        if "k3" not in kwargs:
            kwargs["k3"] = 0
            
        for name in self._param_names:
            self.__setattr__(name, kwargs[name])
    
        self.intrinsics = np.eye(3, np.float64)
        self.distortion = np.zeros((8, 1), np.float64) # rational polynomial
        self.intrinsics[0, 0] = self.fx
        self.intrinsics[1, 1] = self.fy
        self.intrinsics[0, 2] = self.cx
        self.intrinsics[1, 2] = self.cy
        self.distortion[0][0] = self.k1
        self.distortion[1][0] = self.k2
        self.distortion[2][0] = self.p1
        self.distortion[3][0] = self.p2
    
    
    