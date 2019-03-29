import util
import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
from collections import defaultdict
import logging

def camera2lidar(point3d):
    rvec = [1.16916, -1.20437, 1.24619] #[1.17475, -1.15292, 1.120676]
    tvec = [0.234313, -0.347595, -0.17919]#[0.2726317, -0.545845, -0.472]
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec).reshape((3, 1))
    point3d = np.asarray(point3d).reshape((3, 1))
    R, _ = cv2.Rodrigues(rvec)
    new_coord = np.dot(R.T, point3d - tvec)
    return new_coord

def lidar2imu(point3d):
    def euler2rotation(euler_angles):
        """Convert Euler angles to rotation matrix.
    
        Args:
          euler_angles: {list, numpy.ndarray}
              Euler rotation angles in radians, specified as a 1x3 list
              or numpy.array of [yaw, pitch, roll]. The default order for
              Euler angle rotations is "ZYX".
        Returns:
          rotation_matrix: numpy.ndarray
              rotation matrix, return as a 3x3 numpy.ndarray.
        """
        yaw, pitch, roll = euler_angles[0], euler_angles[1], euler_angles[2]
        rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                               [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                               [-np.sin(pitch), 0,
                                np.cos(pitch)]])
        rotation_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                               [0, np.sin(roll), np.cos(roll)]])
        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        return rotation_matrix
    R = euler2rotation([ -0.0447, 0, 0])
#     pdb.set_trace()
    t = np.asarray([0.35, 0.01, 0.18]).reshape((3, 1))
    new_coord = np.dot(R, point3d) + t
    return new_coord

# from mpl_toolkits.mplot3d import Axes3D
# import pptk
calibration_path = "/data/data_arc/truck1_bino_20190301+20190227_calibration/calibration_45_20190302153703.json"
data = util.io.load_json(calibration_path)
camera_model = util.stereo.BinocularModel(**data)
# pdb.set_trace()
def rect2camera(point3d):
    R = camera_model.left_camera_model.R
    point3d = np.asarray(point3d).reshape((3, 1))
    new_coord = np.dot(R.T, point3d)
    return new_coord

path = util.argv[1]

OI = None
if len(util.argv) > 2:
    OI = util.img.imread(util.argv[2])
    
disp = np.load(path)
# disp = cv2.blur(disp, (35, 35))
# disp = util.img.resize(disp, (950, 600))
# disp *= 1920.0 / 950
# disp = util.img.resize(disp, (1920, 960))
img3d = cv2.reprojectImageTo3D(disp, camera_model.Q)
depth = img3d[..., 2]
# print(disp[750, 750], depth[750, 750])
rect = [(900, 600), (1600, 1200)]
# rect = [(683, 663), (1006, 920)]
if util.str.contains(path, "1552626991.157184"):
#     rect = [(880, 621), (1040, 750)] #1552626991.157184
    rect = [(385, 575), (557, 690)]
if util.str.contains(path, "1552627544.329483"):
    rect = [(172, 600), (600, 1000)]
if util.str.contains(path, "1552626812.157"):
    rect = [(1284, 457), (1536, 765)]
#     rect = [(0, 521), (151, 738)]
print(rect)
car_area = img3d[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], ...]
car_depth = car_area[..., 2]
# car_depth = cv2.blur(car_depth, (35, 35))
# util.plt.show_images([car_depth, depth_blur], titles = ["depth", "blurred depth"], share_axis = True)

car_OI = None
edge_mask = np.ones((car_depth.shape))
if OI is not None:
    car_OI = OI[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0], ...]
    edge_mask = cv2.Canny(car_OI, .1, 50) / 255
#     util.plt.imshow("", edge_mask)

util.plt.show_images([depth, OI], show = False, share_axis = True)
class UFS():
    def __init__(self, img, step):
        h, w = img.shape[:2]
        h = int(h / step) + 1
        w = int(w / step) + 1
        self.mask = np.zeros((h, w))
        self.step = step
        self._idx = 1;
    
    def cvt(self, x, y):
        return int(x / step), int(y / step)
    def find_root(self, x, y):
        x, y = self.cvt(x, y)
        return self.mask[y, x]
    
    def next_idx(self):
        idx = self._idx
        self._idx += 1
        return idx
    
    def set_val(self, x, y, val):
        x, y = self.cvt(x, y)
        self.mask[y, x]  = val
        
    def union(self, x1, y1, x2, y2):
        root1 = self.find_root(x1, y1)
        root2 = self.find_root(x2, y2)
        new_root = max([root1, root2])
        if new_root == 0:
            new_root = self.next_idx()
        
        if root1 != new_root:
            if root1 == 0:
                self.set_val(x1, y1, new_root)
            else:
                locs = np.where(self.mask == root1)
                self.mask[locs] = new_root
                
        if root2 != new_root:
            if root2 == 0:
                self.set_val(x2, y2, new_root)
            else:
                locs = np.where(self.mask == root2)
                self.mask[locs] == new_root
        
    def get_max_CN(self):
        max_idx = -1
        max_area = 0
        visited = set()
        h, w = self.mask.shape
        for x in range(w):
            for y in range(h):
                root = self.mask[y, x]
                if root not in visited:
                    area = np.sum(self.mask == root)
                    if max_area < area:
                        max_idx = root
                        max_area = area
        return np.asarray(self.mask == max_idx, dtype = np.int32)
            
step = 5
distance_th = min([0.1 * step, 0.35])
def get_neighbours(x, y, w, h):
    def is_valid(x, y):
        return x < w and y < h
    ns = [(x + step, y), (x + step, y + step), (x, y + step)]
    return [n for n in ns if is_valid(n[0], n[1])]
    
def get_distance(x1, y1, x2, y2, img):
    f1 = img[y1, x1]
    f2 = img[y2, x2]
    dist = f1 - f2
    return np.sqrt(np.dot(dist, np.transpose(dist)))


car_h, car_w = car_area.shape[:2]
ufs = UFS(car_area, step)
for x in range(0, car_w, step):
    for y in range(0, car_h, step):
        ns = get_neighbours(x, y, car_w, car_h)
        for nx, ny in ns:
            distance = get_distance(x, y, nx, ny, car_area)
#             print(distance)
            if distance <= distance_th:
                ufs.union(x, y, nx, ny)

car_mask = ufs.get_max_CN()
car_depth_mask = np.zeros(car_depth.shape)
for x in range(0, car_w, step):
    for y in range(0, car_h, step):
        mx, my = ufs.cvt(x, y)
        if car_mask[my, mx]:
            car_depth_mask[y, x] = car_depth[y, x] 
# car_depth * util.img.resize(car_mask, (car_w, car_h), interpolation = cv2.INTER_NEAREST)
# util.plt.imshow("UFS", car_mask, show = False)

images = [car_depth, car_mask, car_depth_mask]
titles = ["depth", "mask", "depth*mask"]
if car_OI is not None:
    images = [car_OI, edge_mask, car_depth* edge_mask, car_depth* (1 - edge_mask)] + images
    titles = ["Origin", "edge", "edge depth", "non edge depth"] + titles
util.plt.show_images(images, titles = titles, show = False, share_axis = False)

# util.plt.imshow("Depth", car_area[..., 2])
car_depth = car_area[..., 2]
# car_edge = cv2.Canny(np.asarray(car_depth, dtype = np.uint8), .1, 10)
# util.img.imshow("Edge", car_edge)
Xs = car_area[..., 0].ravel()
Ys = car_area[..., 1].ravel()
Zs = car_area[..., 2].ravel()
all_depths = car_depth_mask[np.where(car_depth_mask > 0)]
print(car_depth_mask.max(), all_depths.min())
# util.plt.hist(all_depths)
cell_width_x = 0.05
cell_width_z = 0.05
min_x = 100
max_x = -100
min_z = 100
max_z = -100
car_points = []
for x in range(0, car_w, step):
    for y in range(0, car_h, step):
        mx, my = ufs.cvt(x, y)
        if car_mask[my, mx]:
            px, py, pz = car_area[y, x]
            min_x = min([min_x, px])
            max_x = max([max_x, px])
            min_z = min([min_z, pz])
            max_z = max([max_z, pz])
            car_points.append([px, pz])
cols = int((max_x - min_x) / cell_width_x) + 20
rows = int((max_z - min_z) / cell_width_z) + 20
min_area_rect = cv2.minAreaRect(np.asarray(car_points))
box_points = cv2.boxPoints(min_area_rect).tolist()
cx, cy = min_area_rect[0]
box_w, box_h = min_area_rect[1]
print(box_w, box_h, cx, cy)
# # p3 = car_area[57, 102, :]
# cx = p3[0]
# cy = p3[2]
rect_camera_coord = (cx, 0, cy)
left_camera_coord = rect2camera(rect_camera_coord)
lidar_coord = camera2lidar(left_camera_coord)
imu_coord = lidar2imu(lidar_coord)
print("rect_camera_coord", rect_camera_coord)
print("left_camera_coord", left_camera_coord)
print("lidar_coord", lidar_coord)
print("imu_coord", imu_coord)
box_theta = min_area_rect[2]

def get_grid_coord(x, z):
    gx = int((x - min_x) / cell_width_x) + 10
    gy = int((z - min_z) / cell_width_z) + 10
    gy = rows - gy - 1
    return gx, gy

bev = np.zeros((rows, cols))

for px, pz in car_points:
    grid_x, grid_y = get_grid_coord(px, pz)
    bev[grid_y, grid_x] = 200
    
bev_box_points = np.asarray([get_grid_coord(px, py) for px, py in box_points])
cv2.drawContours(bev,[bev_box_points],0,255,1)

grid_cx, grid_cy = get_grid_coord(cx, cy)
bev[grid_cy, grid_cx] = 255
util.plt.imshow("BEV", bev)

# plt.imshow(np.int32(np.logical_and(car_depth <= max_depth, car_depth >= min_depth)))
# plt.imshow(np.int32(np.logical_and(car_depth < 16, car_depth > 10)))
# plt.show()
# util.plt.imshow("", car_area)
# v = pptk.viewer(img3d)
# v.set(point_size=0.005)
# v.wait()
# print(car_area.shape)