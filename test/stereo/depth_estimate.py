import util
import numpy as np
import pdb
root = "resources/6mm_1.2m/"
calibration_path = root + "calibration_stereo.json"
# left_image_path = root + "0_front_left.png"
# right_image_path = root + "0_front_right.png"

image_path = "/data/stereo/stereo/201912537/origin_%s_1548318633.618236.png"
left_points = [(975, 542), (998, 565), (1030.2, 582.18)] # image_path = "/data/stereo/stereo/201912537/origin_%s_1548318633.618236.png"
right_points = [(898, 620), (922, 642), (954.9, 658.5)]

# depth_gt = 19.3
# image_path = "/data/stereo/stereo/201912538/origin_%s_1548318683.718231.png" # 19.3m
# left_points = [(968.533, 547.699), (1021.33, 588.445)]
# right_points = [(914.373, 624,403), (967.467, 664.378)]

depth_gt = 24.1
image_path = "/data/stereo/stereo/201912538/origin_%s_1548318680.318292.png"
left_points = [(973, 547), (1026, 588)] # image_path = "/data/stereo/stereo/201912537/origin_%s_1548318633.618236.png"
right_points = [(918, 623), (973, 665)]


left_image_path = image_path%("left")
right_image_path = image_path%("right")
left_image = util.img.imread(left_image_path)
right_image = util.img.imread(right_image_path)
# util.plt.show_images([left_image, right_image], axis_off=True, bgr2rgb = True)
data = util.io.load_json(calibration_path)
camera_model = util.stereo.BinocularModel(**data)

for idx, (lp, rp) in enumerate(zip(left_points, right_points)):
    disparity = camera_model.dispairity(lp, rp)
    depth = camera_model.get_depth(lp, rp)
    color = util.img.random_color_3()
    sigma = abs(depth - depth_gt) / depth_gt
    msg = "%d:%.2f"%(idx, depth)
    print("%d:left=%r,right=%r,dispairity=%.2f, depth=%.2f, sigma = %.2f"%(idx, lp, rp, disparity, depth, sigma))
#     util.img.put_text(left_image, msg, lp, scale = 2, color = color, thickness = 2)
    util.img.circle(left_image, lp, r = 3, color = color, border_width = -1)
#     util.img.put_text(right_image, msg, rp, scale = 2, color = color, thickness = 2)
    util.img.circle(right_image, rp, r = 3, color = color, border_width = -1)

left_image = camera_model.left_camera_model.remap(left_image)
right_image = camera_model.right_camera_model.remap(right_image)
# img = np.concatenate([left_image, right_image], axis = 1)
# util.img.imshow("Binocular", img)
util.plt.show_images([left_image, right_image], axis_off=True, bgr2rgb = True)