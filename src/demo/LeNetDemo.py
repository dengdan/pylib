#coding=utf-8
from __future__ import absolute_import

import time
import numpy as np
import theano.tensor as T
import theano

import find_pylib
from model import LeNet
from  data.DataIter import ImageDataIter
import util.io
import util.log
import logging

util.log.init_logger('~/temp/lenet.log');

# 创建Iter的时间越早越好.
logging.info( "creating data iterator...")
val_root_dir = '~/dataset/imagenet_2012_uncompressed/ILSVRC2012_img_val'
val_lst_path = '~/dataset/imagenet_2012_mxnet/val.lst'
train_root_dir = '~/dataset/imagenet_2012_uncompressed/ILSVRC2012_img_train'
train_lst_path = '~/dataset/imagenet_2012_mxnet/train.lst'
batch_size = 100
image_shape = (224, 224)
val_iter = ImageDataIter(root_dir = val_root_dir, lst_path = val_lst_path, image_shape = image_shape, batch_size = batch_size, prefetch = 5, num_threads = 8)
# train_iter = ImageDataIter(root_dir = train_root_dir, lst_path = train_lst_path, image_shape = image_shape, batch_size = batch_size, prefetch = 10, num_threads =8)
train_iter = val_iter

logging.info( 'building theano functions...')
t1 = time.time()

input = T.tensor4()
label = T.ivector()
lr = T.dscalar()
net = LeNet(input, label)
train_alexnet = theano.function(
                                inputs = [input, label, lr], 
                                outputs = [net.loss, net.accuracy], 
                                updates = net.get_updates(lr)
)
val_alexnet = theano.function(
                                inputs = [input, label], 
                                outputs = [net.loss, net.accuracy]
)
get_param_count = theano.function(
                                  inputs = [],
                                  outputs = net.param_count
                                  )

t2 = time.time()
logging.info( 'building finished. %f seconds used' % (t2 - t1))

logging.info( "There are %d parameters in LetNet"%( get_param_count()))
epochs = 20
learning_rate = 0.00001
training_losses = []
training_accuracies = []
val_losses = []
val_accuracies = []
epoch_time = []
dump_path = './model/'
batch_size = 100


for epoch in xrange(epochs):
    iteration = 0
    train_iter.reset()
    t1 = time.time()
    for batch in train_iter:
        data_X = batch.get_data()
        data_y = batch.get_label()
        t2 = time.time()
        io_time = t2 - t1
        training_loss, training_accuracy = train_alexnet(data_X, data_y, learning_rate)
        t1 = time.time()
        training_time = t1 - t2
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        
        logging.info("Epoch %d, Iteration %d: loss = %f, accuracy = %f, training time = %f seconds, io time = %f seconds" % (epoch, iteration, training_loss, training_accuracy, training_time, io_time))
        iteration += 1
    
    # validation
    val_loss = []
    val_accuracy = []
    logging.info( 'validating model...')
    for batch in val_iter:
        data_X = batch.data
        data_y = batch.label
        val_loss_batch, val_accuracy_batch = train_alexnet(data_X, data_y, learning_rate)
        val_loss.append(val_loss_batch)
        val_accuracy.append(val_accuracy_batch)
    val_iter.reset()
    val_losses.append(np.mean(val_loss))
    val_accuracies.append(np.mean(val_accuracy))
    logging.info( "Epoch %d, validation: loss = %f, accuracy = %f"%(epoch, val_losses[-1], val_accuracies[-1]))
    
    # dump data and model
    dump_model = 'Lenet_Epoch%d.model'%(epoch)
    dump_data = 'Lenet_Epoch%d.data'%(epoch)
    t2 = time.time()
    logging.info( 'dumping model and data...')
    util.io.dump(util.io.join_path(dump_path, dump_model), net)
    util.io.dump(util.io.join_path(dump_path, dump_data), {
                                                            "training loss": training_losses,
                                                            "training accuracy": training_accuracies,
                                                            'validation loss': val_losses,
                                                            'validation accuracies': val_accuracies
                                                            })
    t1 = time.time()
    logging.info( 'dumping finished, time used = %f seconds...'%(t1 - t2))
