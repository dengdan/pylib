#encoding=utf-8
"""
read data from icdar dataset.
"""

import util
import evaluate

icdar2015_ch4_training_images = '~/dataset/ICDAR2015/Challenge4/ch4_training_images/'
icdar2015_ch4_training_gt = '~/dataset/ICDAR2015/Challenge4/ch4_training_localization_transcription_gt/'

training_image_name_pattern = 'img_%d.jpg'
training_gt_name_pattern = 'gt_img_%d.txt'


def evaluate_proposal(proposal, idx, img_root_path = icdar2015_ch4_training_images, gt_root_path = icdar2015_ch4_training_gt):
    bboxes, words = get_gt(idx, gt_root_path)
    image = get_image(idx, img_root_path)
    targets = []
    for idx_gt in range(len(bboxes)):
        box, word = bboxes[idx_gt], words[idx_gt]
        score = evaluate.iou(c = proposal, gt = box, I = image)
        if score > 0:
            targets.append((score, box, word)) 
    
    return targets

def get_gt(idx, root_path = icdar2015_ch4_training_gt):
    bboxes = []
    words = []
    gt_path = util.io.join_path(root_path, training_gt_name_pattern%(idx))
    gt = util.io.read_lines(gt_path)
    for line in gt:
        points, word = parse_gt_line(line)
        bboxes.append(points)
        words.append(word)
        
    return bboxes, words
    
    
def get_image(idx, rgb = True, root_path = icdar2015_ch4_training_images):
    mode = util.img.IMREAD_UNCHANGED
    if util.dtype.is_number(idx):
        image_name = training_image_name_pattern%(idx)
    elif util.dtype.is_str(idx):
        image_name = idx
    else:
        raise ValueError
    image_path = util.io.join_path(root_path, image_name)
    image = util.img.imread(image_path, rgb = rgb, mode = mode)
    return image

def get_image_idx(image_name):
    image_name = util.str.remove_all(image_name, 'img_')
    image_name = util.str.remove_all(image_name, '.jpg')
    return int(image_name)
    
def parse_gt_line(line):
    points = [None] * 4
    word = None
    data = line.split(',')
    
    data[0] = data[0].replace('\xef\xbb\xbf', '')
    
    for idx, d in enumerate(data):
        if idx < 8:
            data[idx] = int(d.strip())
       
    word = data[-1].strip()    
    
    for i in xrange(4):
        p = [0] * 2
        p[0] = data[i * 2]
        p[1] = data[i * 2 + 1]
        points[i] = p
        
    return points, word

                    

def show(idx, color, show_text, show_origin, gray):
    image = get_image(idx)
    
    bboxes, words = get_gt(idx)
    
    if show_origin:
        axes = (1, 2)
        ax  = plt.subplot2grid(axes, (0,1))
        ax_image = plt.subplot2grid(axes, (0,0), sharex = ax, sharey = ax)
        ax_image.imshow(image)
    else:
        ax = plt.subplot(111)
    
    if gray:
        image = util.img.rgb2gray(image)
        ax.imshow(image, cmap = 'gray')
    else:
        ax.imshow(image)
    
    # draw bounding box
    for idx_gt in range(len(bboxes)):
        points, word = bboxes[idx_gt], words[idx_gt]
        for i in xrange(4):
            start = i
            end = (i + 1) % 4
            util.plt.line(axis = ax, xy_start = points[start], xy_end = points[end], color = color)
        if show_text:
            x, y = points[1]
            ax.text(x, y, word, verticalalignment='top', horizontalalignment='left', color = color, fontsize=10)
    plt.show()
