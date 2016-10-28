from __future__ import print_function

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
import numpy as np

import theano
import theano.tensor as T

import common_import

import util.nnet.init as init

class RBM(object):
    """Restricted Boltzmann Machine"""
    def __init__(self, input = None, n_visible = 784,n_hidden = 500, W = None, hbias = None,vbias = None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if W is None:
            W = init.random_uniform(n_visible, n_hidden, name = 'W', shared = True)
        
        if hbias is None:
            hbias = init.zeros((n_hidden,), name = 'hbias', shared = True)
        
        if vbias is None:
            vbias = init.zeros((n_visible,), name = 'vbias', shared = True)
        
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        
        self.params = [self.W, self.hbias, self.vbias]
        
    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis = 1)
        return -hidden_term - vbias_term
        
    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
        
    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = util.rand.trng.binominal(size = h1_mean.shape, n = 1, p = h1_mean, dtype = util.dtype.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]
        
        
    def propdown
