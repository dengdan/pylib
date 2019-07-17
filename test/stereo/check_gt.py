import util
path = util.argv[1]
check = None
if len(util.argv > 2):
    check = util.argv[2]
left_path = util.io.join_path(path, "left")
right_path = util.io.join_path(path, "right")
gt_path = util.io.join_path(path, "disparity")

images = util.io.ls(left_path, ".png")
for image_name in images:
    left_image_path = util.io.join_path(left_path, image_name)
    right_image_path = util.io.join_path(right_path, image_name)
    gt_image_path = util.io.join_path(left_path, image_name)
    