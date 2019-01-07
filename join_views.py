import numpy as np
from collections import defaultdict
import util
view_dirs = util.argv[1:]
TIME_DIFF_TH = 0.2
def get_ts(path) :
    ts_str = util.str.find_all(path, "\d+\.\d+")[0]
    return float(ts_str)

def get_frame(path):
    s = util.str.find_all(path, "_\d+_")[0]
    frame = int(util.str.replace_all(s, "_", ""));
    return frame;

def get_planning_img(path, shape, idx):
    ts = get_ts(path);
    new_name = str(idx) + "_Planning_" + util.time.timestamp2str(ts)
    image_data = util.img.imread(path)
    h, w = image_data.shape[:-1]
    pos = (0, 15)
    util.img.put_text(image_data, new_name, pos, 0.5, util.img.COLOR_BGR_RED, 1)
    image_data = util.img.resize(image_data, shape[::-1])
    return image_data

def join_views():
    image_dict = defaultdict(list);
    timestamps = set()
    planning_dict = {}
    num_views = len(view_dirs)
    for view_idx, view_dir in enumerate(view_dirs):
        view_dirs[view_idx] = util.io.get_absolute_path(view_dir)
        image_names = util.io.ls(view_dir, ".jpg")
        for image_name in image_names:
            ts = get_ts(image_name);
            if util.str.contains(image_name, "Planning",  ignore_case = True):
                planning_dict[ts] = view_dir + "/" + image_name
                continue;
            if util.str.contains(image_name, "Fusion"):
                timestamps.add(ts);
            ts = get_ts(image_name)
            image_dict[ts].append(view_dir + "/" + image_name)
    planning_timestamps = list(planning_dict.keys())
    planning_timestamps.sort()
    if planning_timestamps:
        num_views -= 1;
    timestamps = list(timestamps);
    timestamps.sort();
    view_names = [util.io.get_filename(name) for name in view_dirs]
    output_dir = util.io.join_path("~/temp/no-use/", \
                                                util.str.replace_all(view_dirs[0], "/", "_") + util.str.join(view_names, "+"))
    bar = util.ProgressBar(len(image_dict))
    for ts in timestamps:
        bar.move(1)
        images = []
        camera_images = []
        output_path = util.io.join_path(output_dir, str(get_frame(image_dict[ts][0])) +"_" + str(ts) + ".jpg");
        if len(image_dict[ts]) < num_views:
            continue
        if util.io.exists(output_path):
            continue
        for image_path in image_dict[ts]:
            image_name = util.io.get_filename(image_path)
            image = util.img.imread(image_path, rgb = False)
            if util.str.contains(image_name, 'Camera'):
                camera_images.append(image)
            else:
                images.append(image)
        min_diff = 1000;
        min_idx = -1
        for pidx, pts in enumerate(planning_timestamps):
            diff = abs(pts - ts)
            if diff < min_diff:
                min_idx = pidx
                min_diff = diff

        if min_diff < TIME_DIFF_TH:
            images.append(get_planning_img(planning_dict[planning_timestamps[min_idx]], images[0].shape[:-1], min_idx))

        image_data = np.concatenate(images, axis = 1)
        if camera_images:
            h, w = camera_images[0].shape[:-1]
            camera_width = images[0].shape[1]
            if len(camera_images) == 1:
                camera_shape = (image_data.shape[0], camera_width, 3)
                camera_data = np.zeros(camera_shape, dtype = np.uint8)
                camera_height = int(h * (camera_width * 1.0 / w))
                ci = camera_images[0]
                ci = util.img.resize(ci, (camera_width, camera_height))
                camera_data[:ci.shape[0], :ci.shape[1], :] = ci
            else:
                camera_data = np.concatenate(camera_images, axis = 0)
                camera_data = util.img.resize(camera_data, (camera_width, image_data.shape[0]))
    #         image_height = max([camera_data.shape[0], image_data.shape[0]])
    #         camera_data = util.img.resize(camera_data, (camera_data.shape[1], image_height))
    #         image_data = util.img.resize(image_data, (image_data.shape[1], image_height))
            image_data = np.concatenate([camera_data, image_data], axis = 1)
        util.img.imwrite(output_path, image_data)
        #util.plt.show_images(images = images, titles = view_names, save = True, show = True, path = image_path, axis_off = True)

if __name__ == "__main__":
    while True:
        join_views()
