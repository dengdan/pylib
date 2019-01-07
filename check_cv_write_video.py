import util
path = "/tmp/test.mp4"
shape = (500, 500, 3)
fps = 10
with util.video.VideoWriter(path, fps = fps) as video:
    for i in range(1000):
        frame = util.img.black(shape)
        util.img.put_text(frame, str(i), (100, 100), 2, color = util.img.COLOR_WHITE, thickness = 2)
        video.add_frame(frame)
        
fs = util.io.get_file_size(path, "M")
print("actual video filesize = ", fs , "M, expected to 1.7M")
assert fs > 1, "the installed opencv has been installed properly, try: pip install opencv-python"

