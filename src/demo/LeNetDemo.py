#coding=utf-8
from common_import import *

from model import LeNet
from nnet.solver import SimpleGradientDescentSolver

dump_path = '~/temp/lenet'
util.log.init_logger(util.io.join_path(dump_path,'lenet.log'));

batch_size = 100
image_shape = (224, 224)
train_iter, val_iter = get_iter(image_shape = image_shape, batch_size = batch_size, prefetch = 5, num_threads = 8)

net = LeNet()
solver = SimpleGradientDescentSolver(
        batch_size = 100, 
        #epochs = 0.01,
        total_iterations = 20000,
        learning_rate = 0.0001,
        dump_path = dump_path,
        dump_interval = 5000,
        val_interval = 5000, 
        train_iter = train_iter, 
        val_iter = val_iter
        )
solver.fit(model = net)
