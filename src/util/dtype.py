#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import theano
import numpy as np

floatX = theano.config.floatX
int32 = 'int32'

uint8 = np.uint8

def cast(obj, dtype):
    if isinstance(obj, list):
        return np.asarray(obj, dtype = floatX)
    return np.cast[dtype](obj)

