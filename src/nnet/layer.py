#coding=utf-8
'''
Created on 2016年9月27日
@author: dengdan
'''
import theano
import theano.tensor as T
import theano.tensor.signal.pool as pool
import numpy as np

import nnet.init as init
import nnet.fn
import util.dtype as dtype
from util.rand import rng
from util.t import trng
from util.dtype import floatX


class Layer(object):
    def __init__(self, input, name, activation = None):
        self.params = []
        self.name = name
        if activation is None:
            activation = nnet.fn.identity
        self.activation = activation
        if isinstance(input, Layer):
            self.pre = input
            input = input.output
        self.input = input
        
    def get_updates(self, loss, lr):
        return [(p, p - lr * T.grad(loss, p)) for p in self.params]
    

class FullyConnectedLayer(Layer):
    def __init__(self, input, n_in, n_out, activation, W = None, b = None, name = 'fc'):
        Layer.__init__(self, input, name, activation)
        self.n_in = n_in
        self.n_out = n_out
        
        if W is None:
            W_value = init.xavier(n_in = n_in, n_out = n_out)
            W = theano.shared(value = W_value, name = '%s.W_size(%d, %d)'%(name, n_in, n_out))
            self.params.append(W)
        self.W = W
        
        if b is None:
            b_value = np.zeros((n_out,), dtype = dtype.floatX)
            b = theano.shared(value = b_value, name = '%s.b_size(%d,)'%(name, n_out))
            self.params.append(b)
        self.b = b
        
        self.lin_output = T.dot(self.input, self.W) + self.b 
        self.output = self.activation(self.lin_output)

class ConvolutionLayer(Layer):
    def __init__(self, input, filter_shape, stride = (1, 1), activation = T.nnet.relu, padding = 'half', name = 'conv' ):
        Layer.__init__(self, input, name, activation)
        mu, sigma = 0, np.sqrt(2.0/np.prod(filter_shape[1:]))
        W_value = rng.normal(mu, sigma, size = filter_shape)
        W_value = np.asarray(W_value, dtype = floatX)
        self.W = theano.shared(value = W_value, borrow = True)
        b_value = np.zeros((filter_shape[0], ), dtype = floatX)
        self.b = theano.shared(value = b_value, borrow = True)
        
        self.lin_output = T.nnet.conv2d(input = self.input, 
                                    filters = self.W, 
                                    filter_shape = filter_shape, 
                                    border_mode = padding, 
                                    subsample= stride) + self.b.dimshuffle('x', 0, 'x', 'x')
                                    
        if self.activation != None:
            self.output = self.activation(self.lin_output)
        else:
            self.output = self.lin_output
            
        self.params = [self.W, self.b]
        
class MaxPooling(Layer):
    def __init__(self, input, filter_shape = (2, 2), ignore_border = True, stride = None, name = 'max_pooling'):
        Layer.__init__(self, input,  name)
        self.output = pool.pool_2d(input = self.input, ds = filter_shape, ignore_border = ignore_border, st = stride)

class SoftmaxOutputLayerWithLoss(FullyConnectedLayer):
    def __init__(self, input, n_in, n_out, label, name = 'SoftmaxLayerWithLoss'):
        FullyConnectedLayer.__init__(self, input = input,  n_in = n_in, n_out = n_out, activation = T.nnet.softmax, name = name)
        self.predicted = T.argmax(self.output, axis =  1) 
        self.accuracy = T.mean(T.eq(self.predicted, label)) 
        ce = T.log(self.output)[T.arange(0, label.shape[0]), label]
        self.loss = - T.mean(ce)
    

