import util
import numpy as np
import pdb

root = "resources/6mm_1.2m/"
calibration_path = root + "calibration_stereo.json"

depth_gt = 24.1
image_path = "/data/stereo/stereo/201912537/origin_%s_1548318629.318235.png"

left_image_path = image_path%("left")
right_image_path = image_path%("right")
left_image = util.img.imread(left_image_path)
right_image = util.img.imread(right_image_path)
# util.plt.show_images([left_image, right_image], axis_off=True, bgr2rgb = True)
data = util.io.load_json(calibration_path)
camera_model = util.stereo.BinocularModel(**data)
r = 201
util.img.circle(left_image, (0, 0), r = r, color = util.img.COLOR_GREEN, border_width = -1)
util.img.circle(left_image, (1920, 1200), r = r, color = util.img.COLOR_GREEN, border_width = -1)
rect_left_image = camera_model.left_camera_model.remap(left_image)
rect_right_image = camera_model.right_camera_model.remap(right_image)
pdb.set_trace()
util.plt.show_images(images = [left_image, rect_left_image, right_image, rect_right_image], 
             titles = ["left_image", "rect_left_image", "right_image", "rect_right_image"],
             share_axis = True, bgr2rgb = True, 
             shape=(2, 2),
             maximized = True)
