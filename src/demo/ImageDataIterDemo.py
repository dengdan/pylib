#coding=utf-8
'''
Created on 2016年10月12日

@author: dengdan
'''
from data.DataIter import  ImageDataIter , Batch
import time
val_root_dir = '~/dataset/imagenet_2012_uncompressed/ILSVRC2012_img_val'
val_lst_path = '~/dataset/imagenet_2012_mxnet/val.lst'
iter = ImageDataIter(root_dir = val_root_dir, lst_path = val_lst_path, 
                     image_shape = (224, 224), batch_size = 100, random_shuffle = True, center_cropped = True, prefetch = 10, num_threads = 5)
print "Mine:"

iteration = 0
t2 = time.time()
for batch in iter:
    iteration += 1
    t1 = time.time()
    print iteration, batch.data.shape,  t1 - t2
    t2 = time.time()
