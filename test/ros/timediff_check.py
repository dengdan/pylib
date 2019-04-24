import rosbag
import sys
import numpy as np
import pdb

paths = sys.argv[1:-1]
if len(sys.argv) > 2:
    version = sys.argv[-1]
else:
    version = "2"    

diffs = []

def get_ts(msg):
#     pdb.set_trace()
    try:
        return msg.message.header.stamp.to_sec()
    except:
        return msg.message.header.timestamp_sec
def cmp(msg1, msg2):
    return get_ts(msg1) < get_ts(msg2)

if version == "1":
    left_msg_name = "/roadstar/drivers/pylon_camera/camera/frame/front_left/jpg"
    right_msg_name = "/roadstar/drivers/pylon_camera/camera/frame/head_right/jpg"
    lidar_msg_name = "/roadstar/drivers/Pandar40p/Packets"
else:
    left_msg_name = "/roadstar/drivers/pylon_camera/camera/frame/front_left/jpg"
    right_msg_name = "/roadstar/drivers/pylon_camera/camera/frame/front_right/jpg"
    lidar_msg_name = "/roadstar/drivers/lidar/packets/main"

topics = [left_msg_name, right_msg_name, lidar_msg_name]
for path in paths:
    left_camera_msgs = []
    right_camera_msgs = []
    lidar_msgs = []
    bag = rosbag.Bag(path)
    for msg in bag.read_messages(topics):
        if left_msg_name in msg.topic:
            left_camera_msgs.append(msg)
        if right_msg_name in msg.topic:
            right_camera_msgs.append(msg)
        if lidar_msg_name in msg.topic:
            lidar_msgs.append(msg)
#     right_camera_msgs.sort(cmp = cmp)
#     left_camera_msgs.sort(cmp = cmp)
#     lidar_msgs.sort(cmp = cmp)
    def find_nn(cts, msgs, check = lambda t1, t2 : True):
        min_diff = np.Infinity
        for msg in msgs:
            lts = get_ts(msg)
            diff = cts - lts
            if abs(min_diff) > abs(diff) and check(cts, lts):
                min_diff = diff
        return min_diff
    
    for lc_msg, rc_msg, lmsg in zip(left_camera_msgs, right_camera_msgs, lidar_msgs):
        lct = get_ts(lc_msg)
        rct = get_ts(rc_msg)
        print("time_diff between left and right camera is ", lct - rct, 
              "nearest:", find_nn(lct, right_camera_msgs))
        lt = get_ts(lmsg)
        diff = find_nn(lct, lidar_msgs, check= lambda ts1, ts2: ts1 > ts2)
        print("time diff between left camera and lidar", lct - lt, "nearest:", diff)
        if diff < 1:
            diffs.append(diff)
    #     min_diff = find_nn(ct)
    #     if diff != min_diff:
    #         print(diff, min_diff)
freqs, edges = np.histogram(diffs, bins = np.arange(-50, 50, 1.0) / 100)
freqs = np.cumsum(freqs)
for freq, edge in zip(freqs, edges):
    print(freq * 1.0 / len(diffs), edge)
print(len(diffs))
