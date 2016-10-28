#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import matplotlib.pyplot as plt
import numpy as np
import util.img
import util.io
        
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

def plot_solver_data(solver_path):
    data = util.io.load(solver_path)
    training_losses = data.training_losses
    training_accuracies = data.training_accuracies
    val_losses = data.val_losses
    val_accuracies = data.val_accuracies
    plt.figure(solver_path)
    
    n = len(training_losses)
    x = range(n)
    
    plt.plot(x, training_losses, 'r-', label = 'Training Loss')
    
    if len(training_accuracies) > 0:
        plt.plot(x, training_accuracies, 'r--', label = 'Training Accuracy')
    
    if len(val_losses) > 0:
        n = len(val_losses)
        x = range(n)
        plt.plot(x, val_losses, 'g-', label = 'Validation Loss')
        
        if len(val_accuracies) > 0:
            plt.plot(x, val_accuracies, 'g--', label = 'Validation Accuracy')
    plt.legend()
    plt.show()
