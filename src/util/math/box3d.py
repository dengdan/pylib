import numpy as np
from .box2d import Box2d

class Box3d(object):
    def __init__(self, cx, cy, cz, l, h, w, theta, bottom_location = True):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.l = l
        self.h = h
        self.w = w
        self.theta = theta
        if bottom_location:
            self.cy = cy - h / 2.

    @property
    def edges(self):
        corners = self.corners
        bottom_corners = corners[:4]
        top_corners = corners[4:]
        return [(bottom_corners[i - 1], bottom_corners[i]) for i in range(4)] + \
               [(top_corners[i - 1], top_corners[i]) for i in range(4)] + \
               [(bottom_corners[i], top_corners[i]) for i in range(4)]
        
    @property
    def bev_corners(self):
        return Box2d(self.cx, self.cz, self.l, self.w, self.theta).corners
    
    @property
    def center(self):
        return (self.cx, self.cy, self.cz)
    
    @property
    def bev_box2d(self):
        return Box2d(self.cx, self.cz, self.l, self.w, self.theta)
    
    @property
    def corners(self):
        return self._get_corners()
    
    @property
    def half_height(self):
        return self.h / 2
    
    @property
    def volumn(self):
        return self.h * self.w * self.l
    
    def _get_corners(self):
        top_y = self.cy + self.half_height
        bottom_y = self.cy - self.half_height
        top_corners = [(x, top_y, z) for x, z in self.bev_corners]
        bottom_corners = [(x, bottom_y, z) for x, z in self.bev_corners]
        return bottom_corners + top_corners
    
    def project2box2d(self, fn):
        projected_corners = [fn(*c) for c in self.corners]
        xs = [c[0] for c in projected_corners]
        ys = [c[1] for c in projected_corners]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        return (xmin, ymin), (xmax, ymax)
        