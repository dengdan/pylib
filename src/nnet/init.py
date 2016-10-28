#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import numpy as np
from util.rand import rng
import util.dtype as dtype

def random_uniform(n_in, n_out, name = None, shared = False):
    bound = 4.0 * np.sqrt(6.0 / (n_in + n_out))
    val = rng.uniform(low = -bound, high = bound, size  = (n_in, n_out))
    return _ret(val, name = name, shared = shared)

def xavier(n_in, n_out, name = None, shared = False):
    mu, sigma = 0, np.sqrt(2.0/ (n_in + n_out))
    return _ret(rng.normal(mu, sigma, size = (n_in, n_out)), name = name, shared = shared)
    

def zeros(shape, name = None, shared = False):
    val = np.zeros(shape)
    return _ret(val, name = name, sared = shared)    

def _ret(value, name = 'shared_variable', shared = False):
    value = np.asarray(value, dtype = dtype.floatX)
    if shared:
        import theano
        return theano.shared(value, name = name, borrwe = True)
    return value
