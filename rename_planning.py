import util
image_dir = util.io.get_absolute_path(util.argv[1])
target_dir = "/data/fusion/Planning_rename"
if len(util.argv) > 2 :
    target_dir = util.argv[2]
    
image_names = util.io.ls(image_dir, ".jpg")

def get_ts(path) :
    ts_str = util.str.find_all(path, "\d+\.\d+")[0]
    return float(ts_str)

image_names.sort()

pb = util.ProgressBar(len(image_names))
for idx, name in enumerate(image_names):
    src_path = util.io.join_path(image_dir, name)
    ts = get_ts(name);
    new_name = str(idx) + "_Planning_" + util.time.timestamp2str(ts) + "_" + str(ts)+ ".jpg" 
    target_path = util.io.join_path(target_dir, new_name)
    pb.move(1);
    if util.io.exists(target_path):
        continue;
    image_data = util.img.imread(src_path)
    h, w = image_data.shape[:-1]
    pos = (0, int(h * 0.4))
    util.img.put_text(image_data, new_name, pos, 0.5, util.img.COLOR_BGR_RED, 1)
    util.img.imwrite(target_path, image_data);
