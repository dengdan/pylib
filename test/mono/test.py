import util
import pdb
import numpy as np
root = "/Users/dengdan/temp/mono/"
json_path = root + "/calibration_left.json"
images = ["origin_left_1548231668.342947.png", "origin_left_1548231668.942978.png"]
data = util.io.load_json(json_path)
camera_model = util.stereo.CameraModel(**data)
images = [util.img.imread(root + path) for path in images]
images = [camera_model.remap(img) for img in images]

A = [(935, 666), (1175, 690)]
B = [(992, 666), (1247, 690)]
C = [(935, 682), (1175, 712)]
pixels = {"A": A, "B": B, "C": C}
image_coords = {}
for k in pixels.keys():
    ps = [camera_model.undistort_point(p[0], p[1]) for p in pixels[k]]
    pixels[k] = ps
    image_coords[k] = [camera_model.pixel2normalizedimage(*p) for p in ps]

for k in pixels.keys():
    ps = pixels[k]
    for idx, p in enumerate(ps):
        util.img.circle(images[idx], center = p, r = 3, color = util.img.COLOR_GREEN, border_width = -1)
        util.img.put_text(images[idx], text = k, pos = p, scale = 1, color = util.img.COLOR_GREEN, thickness = 1)

# util.plt.show_images(images, bgr2rgb=True, axis_off=True)

def get(name):
    p, c, i = name
    coord = image_coords[p][int(i)]
    if c == "x":
        return coord[0][0]
    elif c == "y":
        return coord[1][0]
    else:
        raise ValueError
matrix = [
    [get("Ax1") - get("Bx1"), get("Ax1") - get("Ax0"), get("Bx0") - get("Bx1"), 0], 
    [get("Ay1") - get("By1"), get("Ay1") - get("Ay0"), get("By0") - get("By1"), 0], 
    [get("Ax1") - get("Cx1"), get("Ax1") - get("Ax0"), 0, get("Cx0") - get("Cx1")], 
    [get("Ay1") - get("Cy1"), get("Ay1") - get("Ay0"), 0, get("Cy0") - get("Cy1")], 
]
matrix = np.asarray(matrix)
print (np.linalg.det(matrix))
print(np.linalg.solve(matrix, np.zeros((4, 1))))