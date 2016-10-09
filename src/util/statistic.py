#coding=utf-8
'''
Created on 2016年10月8日

@author: dengdan
'''
import numpy as np

def D(x):
    if(len(x.shape) > 0):
        x = np.resize(x, np.prod(x.shape))
    return np.var(x)

def E(x):
    if(len(x.shape) > 0):
        x = np.resize(x, np.prod(x.shape))
    return np.average(x)