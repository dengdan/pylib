import numpy as np
from collections import defaultdict
import util
view_dirs = util.argv[1:]
def get_ts(path) :
    ts_str = util.str.find_all(path, "\d+\.\d+")[0]
    return float(ts_str)

image_dict = defaultdict(list);
for view_idx, view_dir in enumerate(view_dirs):
    image_names = util.io.ls(view_dir, ".jpg")
    for image_name in image_names:
        ts = get_ts(image_name)
        image_dict[ts].append(view_dir + "/" + image_name)

timestamps = image_dict.keys();
timestamps.sort();
view_names = [util.io.get_filename(name) for name in view_dirs]
output_dir = "~/temp/no-use/" + util.str.join(view_names, "+");
bar = util.ProgressBar(len(image_dict))
for ts in timestamps:
    bar.move(1)
    images = []
    output_path = util.io.join_path(output_dir, str(ts) + ".jpg");
    if util.io.exists(output_path):
        continue
    for image_path in image_dict[ts]:
        image_name = util.io.get_filename(image_path)
        image = util.img.imread(image_path, rgb = True)
        images.append(image)
    image_data = np.concatenate(images, axis = 1)
    util.img.imwrite(output_path, image_data)
    #util.plt.show_images(images = images, titles = view_names, save = True, show = True, path = image_path, axis_off = True) 
