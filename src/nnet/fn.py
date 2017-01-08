#encoding = utf-8
import theano.tensor as T
import numpy as np
import util
def identity(x):
    return x

def softmax_4d(x_4d):
    """
    x_4d: a 4D tensor:(batch_size,channels, height, width)
    """
    shape = x_4d.shape
    x_3d = x_4d.reshape((shape[0], shape[1], -1))
    m = T.max(x_3d, axis = 1, keepdims = True)
    rebased_x = x_3d - m
    soft_up = T.exp(rebased_x)
    soft_down = T.sum(soft_up, axis = 1, keepdims = True)
    sm = soft_up / soft_down
    return sm.reshape(shape);

def log_softmax_4d(x_4d):
    """
    x_4d: a 4D tensor:(batch_size,channels, height, width)
    """
    shape = x_4d.shape
    x_3d = x_4d.reshape((shape[0], shape[1], -1))
    m = T.max(x_3d, axis = 1, keepdims = True)
    rebased_x = x_3d - m
    lsm_3d = rebased_x - T.log(T.sum(T.exp(rebased_x), axis = 1 , keepdims = True))
    lsm_4d = lsm_3d.reshape(shape)
    return lsm_4d

def cross_entropy_on_log_softmax_result(log_p, target):
    """
    4d operation:[batch_size, channels, height, width]
    """
    dot = log_p * target;
    shape = dot.shape
#     dot_2d = dot.reshape((dot.shape[0], -1))
#     loss =  - T.mean(T.sum(dot_2d, axis = 1))
    loss = - T.sum(dot) / (shape[0]*shape[2] * shape[-1])
    return loss
    
def cross_entropy(p, target):
    """
    shape = (n_examples, ...)
    """
#     p = T.flatten(p, outdim = 1)
#     target = T.flatten(target, outdim = 1)
#     input = target * T.log(p) + (1 - target) * T.log(1 - p)
#     p = p + util.np.TINY
    input = target * T.log(p)
    return T.sum(input =  - input);
#     return T.mean(T.sum(input =  - input, axis = 1))    
    
def mean_square(p, target):
    p = T.flatten(p, outdim = 2)
    target = T.flatten(target, outdim = 2)
    input = T.square(p - target)
    return T.mean(T.sum(input =  input, axis = 1))    

sigmoid = T.nnet.sigmoid
