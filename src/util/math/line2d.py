raise NotImplementedError("not tested")
class Line2d(object):
    def __init__(self, p1 = None, p2 = None, k = None, b = None):
        """
        y = kx + b
        """
        self.is_vertical = False
        self.fixed_value = None
        self.k = k
        self.b = b
        if k is not None:
            assert b is not None
        else:
            assert p1 is not None
            assert p2 is not None
            x1, y1 = p1
            x2, y2 = p2
            if x1 == x2:
                self.is_vertical = True
                self.fixed_value = x1
            else:
                self.k = (y1 - y2) * 1. / (x1 - x2)
                self.b = y1 - self.k * x1
    
    @property
    def is_horizontal(self):
        return self.k != 0
    
    def y(self, x = 0):
        assert not self.is_vertical
        return self.k * x + self.b
    
    def x(self, y):
        assert not self.is_horizontal
        if self.is_vertical:
            return self.fixed_value
        return (y - self.b) / k
    
    def point_distance(self, p):
        x, y = p
        if self.is_vertical:
            return x - self.fixed_value
        if self.is_horizontal:
            return y - self.y()
        prod = x + y * self.k # (x, y) * (1, k)
        proj = abs(prod * 1. / self.k)
        return proj
        
        
