import util
video_path = util.argv[1]
out_dir = util.argv[2]

if not util.io.exists(out_dir):
    util.io.mkdir(out_dir)

reader = util.video.VideoReader(video_path)
for idx, img in enumerate(reader):
    path = "%s/%d.jpg"%(out_dir, idx)
    print("writing to ", path)
    util.img.imwrite(path, img)