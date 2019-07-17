import re
import numpy as np
import os
import cv2 as cv
import util
import pdb

from matplotlib import pyplot as plt

if __name__ == "__main__":
    import sys
    cal_path = sys.argv[1]
    lines = util.io.read_lines(cal_path)
    cal_dict = {}
    for line in lines[2:]:
        key, value = line.split(":")
        value = value.split(" ")
        values = []
        for v in value:
            v = util.str.remove_invisible(v)
            if not v:
                continue
            values.append(float(v))
        cal_dict[key] = values
        
    def get_camera_model(idx):
        idx = str(idx)
        S = cal_dict["S_rect_0" + idx]
        image_height, image_width = S
        size = (int(image_width), int(image_height))
        K = cal_dict["K_0" + idx]
#         fx = K[0]
#         cx = K[2]
#         fy = K[4]
#         cy = K[5]
        D = cal_dict["D_0" + idx]
        k1,k2,p1,p2,k3 = D
        R = cal_dict["R_rect_0" + idx]
        R = np.reshape(R, (3, 3))
        P = cal_dict["P_rect_0" + idx]
        P = np.reshape(P, (3, 4))
        fx = P[0, 0]
        cx = P[0, 2]
        fy = P[1, 1]
        cy = P[1, 2]
        return util.stereo.CameraModel(
            size = size,
            fx = fx, 
            fy = fy, 
            cx = cx, 
            cy = cy,
            k1 = k1, 
            k2 = k2,
            p1 = p1,
            p2 = p2,
            k3 = k3, 
            R = R
        )
        
    left_model = get_camera_model(2)
    right_model = get_camera_model(3)
    R2 = left_model.R
    R3 = right_model.R
    T2 = np.asarray(cal_dict["T_02"])
    T3 = np.asarray(cal_dict["T_03"])
    R = R3 * np.linalg.inv(R2)
#     import pdb
    T = T3 - np.dot(R, T2.T)
    baseline = abs(T[0])
    fx = left_model.fx
    print("baseline =", baseline)
    print("fx =", fx)
    
    root_dir = sys.argv[2]
    dirs = util.io.ls(root_dir)
    values = []
    for d in dirs:
#         if "disp" in d:
        if d == "disp_noc_1":
            dir_path = util.io.join_path(root_dir, d)
            pngs = util.io.ls(dir_path, ".png")
            for png in pngs:
                path = util.io.join_path(dir_path, png)
                img = cv.imread(path, cv.IMREAD_ANYDEPTH)
                print(img.shape)
                img = img / 256.0
                values.extend(img.ravel().tolist())
    count = len(values)
    values = [v for v in values if v]
    print(len(values), len(values) * 1. / count)
    min_disp = min(values)
    max_disp = max(values)
    print("min dist", fx * baseline / min_disp)
    print("max dist", fx * baseline / max_disp)
#     pdb.set_trace()
    print(min_disp, len([v for v in values if v < 5]))
#     plt.hist(values, bins = 20)
#     plt.show()
    ratios, bins = np.histogram(values, bins = 50, density = True) 
    print(len([v for v in values if v >=7 and v <= 60]) * 1. / len(values))
    print(fx * baseline / 7, fx * baseline / 60)
#     for r, b in zip(ratios, bins):
#         print(r, b)
        
