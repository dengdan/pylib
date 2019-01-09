import cv2
import numpy as np

import util

data = util.io.load_json("resources/calibration.json")
camera_model = util.stereo.CameraModel(data)
board = util.stereo.CalibrationBoard(n_rows = 6, n_cols= 8)
obj_points = board.get_object_points()
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

img = cv2.imread("resources/1.png")
# import  pdb
# pdb.set_trace()

def draw(img, corners, img_pts, idx = 0):
#     corner = tuple(corners[0 + idx].ravel())
#     img = cv2.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
#     img = cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
#     img = cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
    for i in range(np.shape(img_pts)[0]):
#         img = cv2.line(img, tuple(corners[i].ravel()), tuple(img_pts[i].ravel()), (255, 255, 0), 5)
#         util.img.circle(img, img_pts[i].ravel(), r = 11, color = util.img.COLOR_BGR_RED)
        right = i + 1
        down = i + board.n_cols
        if right < len(img_pts) and right // board.n_cols == i // board.n_cols:
            img = cv2.line(img, tuple(img_pts[i].ravel()), tuple(img_pts[right].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, tuple(corners[i].ravel()), tuple(corners[right].ravel()), (255, 0, 0), 5)
        if down < len(img_pts):
            img = cv2.line(img, tuple(img_pts[i].ravel()), tuple(img_pts[down].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, tuple(corners[i].ravel()), tuple(corners[down].ravel()), (255, 0, 0), 5)
    return img
# img = board.draw_corners(img.copy(), corners = corners)
ok, corners = board.find_corners(img)
_, R, t, inliners = cv2.solvePnPRansac(obj_points, corners, camera_model.intrinsics, 
                                    camera_model.distortion)
obj_points[:, :, -1] = -3
img_pts, jac = cv2.projectPoints(obj_points, R, t, camera_model.intrinsics, 
                                 camera_model.distortion)
img = draw(img, corners, img_pts, idx = 3)
util.img.imshow("", img)
