#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
import numpy as np
import theano.tensor as T
rng = np.random.RandomState(1234)
trng = T.shared_randomstreams.RandomStreams(rng.randint((2 ** 30)))