#coding=utf-8
'''
Created on 2016年10月12日
参考jxh的代码, 利用consumer-producer模型实现I/O
@author: dengdan
'''
class DataIter(object):
    def __init__(self, lst_path, image_shape, random_shuffle = True, center_cropped = True, pattern = ''):
        self.lst_path = lst_path
        self.image_shape = image_shape
        self.random_shuffle = random_shuffle
        self.center_cropped = center_cropped
        self.pattern = pattern
        
        