import util
import pdb
import numpy as np
root = "../stereo/resources/binocular/"
json_path = root + "left.json"
image_path = root + "left.png"
data = util.io.load_json(json_path)
camera_model = util.stereo.CameraModel(**data)
camera_model.set_alpha(1)
image_data = util.img.imread(image_path)
image_data = util.img.black(image_data.shape)
h, w, _ = image_data.shape

step = 300
thickness = 3
color = util.img.COLOR_GREEN
for x in range(0, w, step):
    top = (x, 0)
    btn = (x, h)
    util.img.line(image_data, top, btn, color, thickness)
for y in range(0, h, step):
    left = (0, y)
    right = (w, y)
    util.img.line(image_data, left, right, color, thickness)
    
undistored = camera_model.remap(image_data)
util.plt.show_images(images = [image_data, undistored], titles = ["origin", "undistored"], 
                     share_axis = True, bgr2rgb = True)