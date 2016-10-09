#coding=utf-8
'''
Created on 2016年10月7日

@author: dengdan
'''
import mxnet as mx

imagenet2012_data_dir = '/home/dengdan/dataset/imagenet_2012_mxnet/'
train_dataset = 'train.rec'
val_dataset = 'val.rec'
data_shape = (3, 224, 224)
def get_imagenet_2012_train_data(batch_size = 100, data_path = imagenet2012_data_dir, rand_crop  = True, rand_mirror = True ):
    train = mx.io.ImageRecordIter(
        path_imgrec = data_path + train_dataset,
#         mean_r      = 123.68,
#         mean_g      = 116.779,
#         mean_b      = 103.939,
        rand_crop   = rand_crop,
        rand_mirror = rand_mirror,
        data_shape  = data_shape,
        batch_size  = batch_size,
        )
    return train


def get_imagenet_2012_val_data(batch_size = 100, data_path = imagenet2012_data_dir , rand_crop  = False, rand_mirror = False):
    val = mx.io.ImageRecordIter(
        path_imgrec = data_path+ val_dataset,
#         mean_r      = 123.68,
#         mean_g      = 116.779,
#         mean_b      = 103.939,
        rand_crop   = rand_crop,
        rand_mirror = rand_mirror,
        data_shape  = data_shape,
        batch_size  = batch_size)
    return val

