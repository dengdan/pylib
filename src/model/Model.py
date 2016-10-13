#coding=utf-8
'''
Created on 2016-10-13 14:26:12

@author: dengdan
'''
import theano.tensor as T
import util.dtype
class Model(object):
    def __init__(self):
        self.params = []
        self.layers = []
        
    def touch_params(self):
        if len(self.params) == 0:
            for l in self.layers:
                self.params.extend(l.params)
        param_count = 0
        for p in self.params:
            param_count += T.prod(p.shape)
        self.param_count = param_count
    
    def get_updates(self, lr):    
#         make sure lr has the same type with params
        lr = T.cast(lr, util.dtype.floatX)
        updates = [(p, p - lr * T.grad(self.loss, p)) for p in self.params]
        return updates        
        
