import util
import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
calibration_path = "/data/data_arc/truck1_bino_20190301+20190227_calibration/calibration_45_20190302153703.json"
data = util.io.load_json(calibration_path)
camera_model = util.stereo.BinocularModel(**data)

path = util.argv[1]
data = np.load(path)
h, w = data.shape
data = util.img.resize(data, (w / 2, h / 2))
data *= 2
disp = util.img.resize(data, (w, h))
img3d = cv2.reprojectImageTo3D(disp, camera_model.Q)

Xs = img3d[..., 0].ravel()
Ys = img3d[..., 1].ravel()
Zs = img3d[..., 2].ravel()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xs, Ys, Zs, c='r', marker='o')
ax.set_xlabel('Y')
ax.set_ylabel('Z')
ax.set_zlabel('X')
plt.show()

depth = img3d[..., 2]
# pdb.set_trace()

# util.plt.imshow("", disp)
rect = [(920, 700), (1600, 1200)]
car_area = depth[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
# util.plt.imshow("", car_area)
# v = pptk.viewer(img3d)
# v.set(point_size=0.005)

print(car_area.shape)