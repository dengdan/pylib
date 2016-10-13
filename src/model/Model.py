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
        self.input = T.tensor4()
        self.label = T.ivector()
        self.lr = T.cast(T.dscalar(), util.dtype.floatX)
        self.name = name        
        
    def touch_params(self):
        """generate param information for the model
        make sure it is called before or at the end of initialization"""
        if len(self.params) == 0:
            for l in self.layers:
                self.params.extend(l.params)
        param_count = 0
        for p in self.params:
            param_count += T.prod(p.shape)
        self.param_count = param_count
    
    def get_updates(self): 
        updates = [(p, p - self.lr * T.grad(self.loss, p)) for p in self.params]
        return updates        
        
        
    def get_training_fn(self):
        """
        the returned training function accepts three parameters:
            input: 4-D training data batch, with shape = (batch_size, channels, height, width)
            label: a vector of integers as the label
            lr: learning rate
        """
        t1 = time.time()
        logging.info('building the training function of %s...' %(self.name))
        train = theano.function(
                inputs = [self.input, self.label, self.lr], 
                outputs = [self.loss, self.accuracy], 
                updates = self.get_updates()
        )
        t2 = time.time()
        logging.info("building finished, using %d seconds."%(t2 - t1))
        return train
        
    def get_val_fn(self):
        t1 = time.time()
        logging.info('building the val function of %s...' %(self.name))
        val = theano.function(
                                        inputs = [self.input, self.label], 
                                        outputs = [self.loss, self.accuracy]
        )
        t2 = time.time()
        logging.info("building finished, using %d seconds."%(t2 - t1))
        return val
    
    def get_param_count_fn(self):
        get_param_count = theano.function(
                                          inputs = [],
                                          outputs = self.param_count
                                          )        
        return get_param_count
