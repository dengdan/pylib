import util
import pdb
import numpy as np
root = "../stereo/resources/binocular/"
json_path = root + "left.json"
data = util.io.load_json(json_path)
camera_model = util.stereo.CameraModel(**data)
print(camera_model.fov())
print(camera_model.visible_range(2.3))

print("horizontal and vertical fov of 25mm camera is ", util.stereo.fov(4000, 1920), "degrees")
