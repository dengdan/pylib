#coding=utf-8
'''
@author: dengdan
'''
#from common_import import *
from  data.DataIter import ImageDataIter
import logging

def get_iter(batch_size = 100, image_shape = (224, 224), prefetch = 5, num_threads = 8):
    logging.info( "creating data iterator...")
    val_root_dir = '~/dataset/imagenet_2012_uncompressed/ILSVRC2012_img_val'
    val_lst_path = '~/dataset/imagenet_2012_mxnet/val.lst'
    train_root_dir = '~/dataset/imagenet_2012_uncompressed/ILSVRC2012_img_train'
    train_lst_path = '~/dataset/imagenet_2012_mxnet/train.lst'
    val_iter = ImageDataIter(root_dir = val_root_dir, lst_path = val_lst_path, image_shape = image_shape, batch_size = batch_size, prefetch = prefetch, num_threads = num_threads)
    train_iter = ImageDataIter(root_dir = train_root_dir, lst_path = train_lst_path, image_shape = image_shape, batch_size = batch_size, prefetch = prefetch, num_threads = num_threads)
    return train_iter, val_iter
            
