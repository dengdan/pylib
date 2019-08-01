from enum import Enum

class Color(Enum):
    def __init__(self, r = None, g = None, b = None, rgb = None):
        if rgb:
            b = rgb & 0xff
            g = (rgb >> 8) & 0xff
            r = (rgb >> 16) & 0xff
        self.r = r
        self.g = g
        self.b = b
        
    def bgr(self):
        return (self.b, self.g, self.r)
    
    def rgb(self):
        return (self.r, self.g, self.b)
    
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Green = (0, 255, 0)
    Red = (255, 0, 0)
    Yellow = (255, 255, 0)
    Gray = (47, 79, 79)
    Pink = (255, 192, 203)
    Blue = (0, 0, 255)
    Purple = (0xa0, 0x20, 0xf0, 0xa020f0)
    IndianRed = (None, None, None, 0xb0171f)
    MeiRed = (None, None, None, 0xdda0dd)
    DeepRed = (None, None, None, 0xff00ff)
    GhostWhite = (None, None, None, 0xf8f8ff)