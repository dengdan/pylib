#encoding = utf-8
import theano.tensor as T

def identity(x):
    return x


def cross_entropy(p, target):
    """
    shape = (n_examples, ...)
    """
    p = T.flatten(p, outdim = 2)
    target = T.flatten(target, outdim = 2)
    input = target * T.log(p) + (1 - target) * T.log(1 - p)
    return T.mean(T.sum(input =  - input, axis = 1))    
    
def mean_square(p, target):
    p = T.flatten(p, outdim = 2)
    target = T.flatten(target, outdim = 2)
    input = T.square(p - target)
    return T.mean(T.sum(input =  input, axis = 1))    

sigmoid = T.nnet.sigmoid
