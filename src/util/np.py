import numpy as np
import copy

concat = np.concatenate

def norm1(v):
    return np.sqrt(norm2(v))

def norm2(v):
    return np.sum(v ** 2)

def cos_dist(v1, v2):
    length1 = length(v1)
    length2 = length(v2)
    return np.dot(v1, v2) / (length1 * length2)

def eu_dist(v1, v2):
    v = v1 - v2
    return norm1(v)

def chi_squared_dist(f1, f2):
    dist = 0
    for ff1, ff2 in zip(f1, f2):
        if ff1 + ff2 == 0:# color feature values are supposed to be non-negative. If this case happened, it means both ne and de are 0s 
            continue;
        dist += (ff1 - ff2) ** 2 * 1.0/ (ff1 + ff2) 
    return np.sqrt(dist)

def flatten(arr, ndim = 1):
    """
    flatten an multi-dimensional array to a certain degree.
    ndim: the number of dimensions after flatten
    """
    arr = np.asarray(arr)
    dims = len(arr.shape)
    shape = [np.prod(arr.shape[0: dims + 1 - ndim])]
    shape.extend(arr.shape[dims + 1 - ndim: dims])
    return np.reshape(arr, shape)

def arcsin(sins, xs = None):
    """
    cal arcsin.
    xs: if this parameter is provided, the returned arcsins will be within [0, 2*pi)
    otherwise the default [-pi/2, pi/2]
    """
    arcs = np.arcsin(sins);
    if xs != None:
        xs = np.asarray(xs)
        sins = np.asarray(sins)
        # if x > 0, then the corresponding mask value is  -1. The resulting angle unchanged: v = 0 - (-v) = v.  else, v = pi - v
        add_pi = xs < 0
        pi_mask = add_pi * np.pi
        # 0 --> 1, 1 --> -1
        arc_mask = 2 * add_pi - 1
        arcs = pi_mask - arcs * arc_mask

        # if x >= 0 and sin < 0, v = 2*pi + v
        add_2_pi = (xs >= 0) * (sins < 0)
        pi_mask = add_2_pi * 2 * np.pi
        arcs = pi_mask + arcs 
    return arcs
    
def sin(ys = None, lengths = None, xs = None, angles = None):
    """
    calculate sin with multiple kinds of parameters
    """
    if not angles is None:
        return np.sin(angles)
    
    if ys is None:
        raise ValueError('ys must be provided when "angles" is None ')

    if lengths is None:
        if xs is None:
            raise ValueError('xs must be provided when "lengths" is None ')
        lengths = np.sqrt(xs ** 2 + ys ** 2)
    
    if not np.iterable(lengths):
        sins = ys / lengths if lengths > 0 else 0
    else:
        lengths = np.asarray(lengths)
        shape = lengths.shape
        ys = flatten(ys)
        lengths = flatten(lengths)                
        sins = [y / length if length > 0 else 0 for (y, length) in zip(ys, lengths)]
        sins = np.reshape(sins, shape)
    return sins

def sum_all(m):
    """
    sum up all the elements in a multi-dimension array
    """
    return sum(flatten(m))
    
    
def clone(obj, deep = False):
    if not deep:
        return copy.copy(obj)
    return copy.deepcopy(obj)

def empty_list(length, etype):
    empty_list = [None] * length
    for i in xrange(length):
        if etype == list:
            empty_list[i] = []
        else:
            raise NotImplementedError
    
    return empty_list
            
        
        
