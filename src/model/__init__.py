#coding=utf-8
'''
Created on 2016-10-13 14:26:12
@author: dengdan
'''
import logging
import time

import theano
import theano.tensor as T

import util

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
        
    
    @util.dec.print_calling    
    def get_param_count_fn(self):
        get_param_count = theano.function(
                                          inputs = [],
                                          outputs = self.param_count
                                          )        
        return get_param_count
        
    @util.dec.print_calling
    def get_predict_fn(self, model = None):
        if model is None:
            model = self
        fn = theano.function(
            inputs = [model.input],
            outputs = [model.predicted, model.output]
        )
        return fn
    
    @util.dec.print_calling
    def get_accuracy_fn(self, model = None):
        if model is None:
            model = self
            
        fn = theano.function(
            inputs = [model.input, model.label],
            outputs = [model.accuracy, model.loss, model.predicted, model.output]
        )
        return fn
    
    def get_sensitivity_fn(self):
        fn = theano.function(
                        inputs = [self.input, self.label],
                        outputs = [T.grad(self.loss, layer.lin_output) for layer in self.layers]
            )
        return fn
        
    def get_output_fn(self):
        fn = theano.function(
                         inputs = [self.input],
                         outputs = [layer.output for layer in self.layers]
                             )
        return fn
    
    def get_grad_fn(self):
        update_value = [T.grad(self.loss, p) for p in self.params]
        fn = theano.function(
                             inputs = [self.input, self.label],
                             outputs = update_value
                             )
        return fn
    
    def get_weight_values(self):
        weights = [layer.params[0].get_value() for layer in self.layers]
        return weights
    
def get_predict_fn(model):
    m = Model('')
    return m.get_predict_fn(model)
    
def get_accuracy_fn(model):
    m = Model('')
    return m.get_accuracy_fn(model)
