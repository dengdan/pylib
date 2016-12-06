#encoding=utf-8
"""
read data from icdar dataset.
"""

import util
import evaluate

from config import training_image_name_pattern, training_gt_name_pattern

GREEN = 'green'


def evaluate_proposal(proposal, idx):
    bboxes, words = get_gt(idx)
    image = get_image(idx)
    targets = []
    for idx_gt in range(len(bboxes)):
        box, word = bboxes[idx_gt], words[idx_gt]
        score = evaluate.iou(c = proposal, gt = box, I = image)
        if score > 0:
            targets.append((score, box, word)) 
    
    return targets

def get_gt(idx):
    bboxes = []
    words = []
    gt = util.io.read_lines(training_gt_name_pattern%(idx))
    for line in gt:
        points, word = parse_gt_line(line)
        bboxes.append(points)
        words.append(word)
        
    return bboxes, words
    
    
def get_image(idx):
    rgb = True
    mode = util.img.IMREAD_UNCHANGED
    image = util.img.imread(training_image_name_pattern%(idx), rgb = rgb, mode = mode)
    return image

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
