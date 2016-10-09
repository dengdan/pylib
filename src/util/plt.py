#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import matplotlib.pyplot as plt
import cPickle as pkl
import numpy as np
import util.img
import util.io
def plot_train_val_loss_acc(path):
    """
    data format in pkl: [(name, y_values), ...]
    the length of all y_values should be the same.
    """
    with open(path, 'r') as f:
        data = pkl.load(f)
        
    x_values = range(len(data[0][1]))
    for name, y_values in data:
        plt.plot(x_values, y_values, label = name)
    plt.legend()
    
    plt.show()
        
def hist(x, title = None, save_path = '/home/dengdan/images/', show = False, block = False, bin_count = 100, bins = None):    
    if len(x.shape) > 1:
        x = np.reshape(x, np.prod(x.shape))
    if bins == None:
        bins = np.linspace(start = min(x), stop = max(x), num = bin_count, endpoint = True, retstep = False)
    plt.figure(num = title)
    plt.hist(x, bins)
    util.io.mkdir(save_path);
    path = save_path + title + '.png'
    plt.savefig(path)
    plt.close()
    if show:
#         plt.show(block = block)
        util.img.imshow(title, path, block = block)        
        