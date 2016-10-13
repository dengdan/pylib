#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import numpy as np

float32 = 'float32'
floatX = float32
int32 = 'int32'
uint8 = 'uint8'
string = 'str'
int = 'int'

def cast(obj, dtype):
    if isinstance(obj, list):
        return np.asarray(obj, dtype = floatX)
    return np.cast[dtype](obj)

