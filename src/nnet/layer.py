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
import util.dtype as dtype
from util.rand import rng, trng
from util.dtype import floatX

def add_noise(input, noise_level):
    noise = trng.binomial(size = input.shape, n = 1, p = 1 - noise_level)
    return noise * input

def cross_entropy(p, target):
    """
    only 2-D supported by now
    shape = (n_examples, n_categories)
    """
    input = target * T.log(p) + (1 - target) * T.log(1 - p)
    return T.mean(T.sum(input =  - input, axis = 1))

class Layer(object):
    def __init__(self, input, name):
        self.params = []
        self.name = name
        if isinstance(input, Layer):
            self.pre = input
            input = input.output
        self.input = input
        
    def get_updates(self, loss, lr):
        return [(p, p - lr * T.grad(loss, p)) for p in self.params]
    

class FullyConnectedLayer(Layer):
    def __init__(self, input, n_in, n_out, activation, W = None, b = None, name = 'fc'):
        Layer.__init__(self, input, name)
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        
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
        Layer.__init__(self, input, name)
        mu, sigma = 0, np.sqrt(2.0/np.prod(filter_shape[1:]))
        W_value = rng.normal(mu, sigma, size = filter_shape)
        W_value = np.asarray(W_value, dtype = floatX)
        self.W = theano.shared(value = W_value, borrow = True)
        self.activation = activation
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
    
class AutoEncoder(object):
    """
    An AutoEncoder consists of three layers: input layer, hidden layer, and reconstruct layer.
    The hidden layer is a fully  connected layer: hfc, with params = [W, b], and the output of the hidden layer is: hfc.activation(input * W + b)
    The reconstruct layer is also a fully connected layer: rfc, with its own W or rfc.W = hfc.W.T
    """
    def __init__(self, input, n_visible, n_hidden, noise_level = None, share_weight = True):
        if noise_level is not None:
            input = add_noise(input, noise_level = noise_level)
            
        hidden_layer = FullyConnectedLayer(input = input, n_in = n_visible, n_out = n_hidden, activation = T.nnet.softmax, name='HiddenLayer')
        
        self.hidden_layer = hidden_layer
        
        self.output = self.hidden_layer.output
        
        reconstruct_W = None
        if share_weight:
            reconstruct_W = self.hidden_layer.W.T
        reconstruct_layer = FullyConnectedLayer(input = self.output, n_in = n_hidden, n_out = n_visible, activation = T.nnet.sigmoid, W = reconstruct_W)
        self.reconstruct_layer = reconstruct_layer
        self.reconstructed = self.reconstruct_layer.output
        self.loss = cross_entropy(self.reconstructed, input)
        
        
