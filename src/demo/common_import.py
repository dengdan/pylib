from __future__ import absolute_import
try:
    import util
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    p = os.path.join(curr_path, "../")
    sys.path.append(p)
    import util, nnet

import time

import numpy as np
import theano.tensor as T
import theano

import util.io
import util.log
import logging

from nnet.solver import SGDSolver

from imagenet2012_iter import get_iter
