#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import numpy as np
from util.rand import rng
def random_uniform(n_in, n_out):
    bound = 4.0 * np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(low = -bound, high = bound, size  = (n_in, n_out))

def xavier(n_in, n_out):
    mu, sigma = 0, np.sqrt(2.0/ (n_in + n_out))
    return rng.normal(mu, sigma, size = (n_in, n_out))