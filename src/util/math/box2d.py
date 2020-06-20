import numpy as np
class Box2d(object):
    def __init__(self, cx, cy, w, h, theta):
        """
        w : along x axis
        h : along y axis
        theta : the angle between w and x
        """
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.theta = theta
        self._corners = self._get_corners()
        
    @property
    def corners(self):
        return self._corners
    
    @property
    def center(self):
        return self.cx, self.cy
    
    @property
    def edges(self):
        return [(self.corners[i], self.corners[i + 1]) for i in range(-1, 3)]
    
    @property
    def half_width(self):
        return self.w / 2.
    
    def to_polygon(self):
        from shapely.geometry import Polygon
        return Polygon(self.corners)
        
    def area(self):
        if self.theta == 0:
            return self.w * self.h
        return self.to_polygon().area
    
    def intersection(self, other):
        poly1 = self.to_polygon()
        poly2 = other.to_polygon()
        area1 = poly1.area
        area2 = poly2.area
        inter = poly1.intersection(poly2).area
        return inter
    
    def iou(self, other, criteria = -1):
        """
        criteria:
            -1: inter / union
            0 : inter / self.area
            1 : inter / other.area
        """
        assert isinstance(other, Box2d)
        poly1 = self.to_polygon()
        poly2 = other.to_polygon()
        area1 = poly1.area
        area2 = poly2.area
        inter = poly1.intersection(poly2).area
        if criteria == -1:
            union = area1 + area2 - inter
            return inter / union
        elif criteria == 0:
            return inter / area1
        elif criteria == 1:
            return inter / area2
        else:
            raise ValueError("invalid criteria: %d"%(criteria))
    
    @property
    def half_height(self):
        return self.h / 2.
    
    def _get_corners(self):
        """
      1 ------ 0
        |    |  ---> heading
      2 ------ 3
        
        """
        sin = np.sin(self.theta)
        cos = np.cos(self.theta)
        center = np.asarray((self.cx, self.cy))
        # cal rotation matrix
        rot = np.array([[cos,  -sin],
                        [sin, cos]])
        corners = np.asarray([[self.half_width, self.half_height],
                              [-self.half_width, self.half_height],
                              [-self.half_width, -self.half_height],
                              [self.half_width, -self.half_height]])
        corners = np.matmul(rot, corners.T) + center.reshape((2, 1))
#         u = np.asarray((self.half_width * cos, self.half_width * sin))
#         v = np.asarray((-self.half_height * sin, self.half_height * cos))
#         corners = [center + u + v,  \
#                    center - u + v, \
#                    center - u - v,  \
#                    center + u - v]
        return [c.tolist() for c in corners.T]
    
if __name__ == "__main__":
    box1 = Box2d(0, 0, 20, 20, 0)
    box2 = Box2d(10, 0, 10, 10, 0)
    import util
    util.test.assert_almost_equal(box1.iou(box2, criteria= -1), 50. / 450)
    util.test.assert_almost_equal(box1.iou(box2, criteria= 0), 50. / 400)
    util.test.assert_almost_equal(box1.iou(box2, criteria= 1), 50. / 100)
    util.test.assert_equal(box1.area(), 400)