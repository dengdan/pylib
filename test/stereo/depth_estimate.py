import util
import numpy as np
import pdb
root = "resources/6mm_1.2m/"
calibration_path = root + "calibration_stereo.json"
left_image_path = root + "0_front_left.png"
right_image_path = root + "0_front_right.png"
left_image = util.img.imread(left_image_path)
right_image = util.img.imread(right_image_path)
# util.plt.show_images([left_image, right_image], axis_off=True)
left_points = [(1518, 207)]
right_points = [(314, 289)]
data = util.io.load_json(calibration_path)
camera_model = util.stereo.BinocularModel(**data)

for idx, (lp, rp) in enumerate(zip(left_points, right_points)):
    depth = camera_model.get_depth(lp, rp)
    color = util.img.random_color_3()
    msg = "%d:%.2f"%(idx, depth)
    print("%d:left=%r,right=%r,depth=%.2f"%(idx, lp, rp, depth))
    util.img.put_text(left_image, msg, lp, scale = 2, color = color, thickness = 2)
    util.img.circle(left_image, lp, r = 3, color = color, border_width = -1)
    util.img.put_text(right_image, msg, rp, scale = 2, color = color, thickness = 2)
    util.img.circle(right_image, rp, r = 3, color = color, border_width = -1)

left_image = camera_model.left_camera_model.remap(left_image)
right_image = camera_model.right_camera_model.remap(right_image)
img = np.concatenate([left_image, right_image], axis = 1)
util.img.imshow("Binocular", img)
