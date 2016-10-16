#coding=utf-8
'''
Created on 2016-10-7
@author: dengdan
'''
from common_import import *

from model import AlexNet
from nnet.solver import MomentumGradientDescentSolver

dump_path = '~/temp/results/momentum_alexnet'
util.log.init_logger(util.io.join_path(dump_path,'momentum_alexnet.log'));

batch_size = 128
image_shape = (224, 224)
train_iter, val_iter = get_iter(image_shape = image_shape, batch_size = batch_size, prefetch = 5, num_threads = 8)

net = util.io.join_path(dump_path, 'AlexNet2_Iteration_9999.model')
logging.info('reloading model from file: %s'%(net))
net = util.io.load(net)
logging.info('reloading finished.')
solver = MomentumGradientDescentSolver(
        epochs = 50,
        momentum = 0.9,
        decay = 0.0005,
#         total_iterations = 20,
        learning_rate = 0.001,
        dump_path = dump_path,
        dump_interval = 5000,
        val_interval = 5000, 
        train_iter = train_iter,
        val_iter = val_iter
        )
# solver = util.io.join_path(dump_path, 'AlexNet2_Iteration_9999.solver.data')
# logging.info('reload learning data from file: %s' %(solver))
# solver = util.io.load(solver)
# logging.info('reloading finished.')
# solver.epochs = 50
# solver.learning_rate = 0.001
# solver.dump_path = dump_path
# solver.train_iter = train_iter
# solver.val_iter = val_iter

logging.info('start training...')
solver.fit(net, last_stop_iteration = 9999)

