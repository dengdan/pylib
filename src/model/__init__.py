#coding=utf-8
'''
Created on 2016-10-13 14:26:12
@author: dengdan
'''
import logging
import time
import numpy as np
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
        make sure it is called before or at the end of initialization
        """
        self.params_to_be_updated = []
        self.lr_mul = []
        self.params = []
        for layer_name in self.layers:
            l = self.layers[layer_name]
            self.params.extend(l.params)
            if l.lr_mul > 0:
                logging.info('layer %s is going to be trained, lr_mul = %f.'%(l.name, l.lr_mul))
                for p in l.params:
                    logging.info('\t\t %s', p.name)
                self.params_to_be_updated.extend(l.params)
                self.lr_mul.extend([l.lr_mul] * len(l.params))
            else:
                logging.info('layer %s is not going to be trained.'%l.name)
        param_count = 0
        for p in self.params_to_be_updated:
            param_count += T.prod(p.shape)
        self.param_count = param_count
        
    def init_params(self, path):   
        params = util.io.load(path)
        for layer_name in self.layers:
            layer = self.layers[layer_name]
            key_pattern = layer.name + '_%s'
            weight_key = key_pattern%('weight')
            bias_key = key_pattern%('bias')
            if weight_key in params:
                logging.info('initializing weights and bias of layer %s'%(layer.name))
                w = np.asarray(params[weight_key].asnumpy(), dtype = np.float32)
                b = np.asarray(params[bias_key].asnumpy(), dtype =  np.float32)
                layer.W.set_value(w)
                layer.b.set_value(b)
         
    
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
