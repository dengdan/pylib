#coding=utf-8
'''
Created on 2016-10-13 14:26:12

@author: dengdan
'''
import logging
import time

import theano
import theano.tensor as T

import util.dtype
class Model(object):
    def __init__(self, name):
        self.params = []
        self.layers = []
        self.params_to_be_updated = []
        self.input = T.tensor4()
        self.label = T.ivector()
        self.name = name        
        
    def touch_params(self):
        """generate param information for the model
        make sure it is called before or at the end of initialization"""
        if len(self.params) == 0:
            for l in self.layers:
                self.params.extend(l.params)
        if len(self.params_to_be_updated) == 0:
            self.params_to_be_updated = self.params
        param_count = 0
        for p in self.params_to_be_updated:
            param_count += T.prod(p.shape)
        self.param_count = param_count
        
        
    def get_param_count_fn(self):
        get_param_count = theano.function(
                                          inputs = [],
                                          outputs = self.param_count
                                          )        
        return get_param_count
