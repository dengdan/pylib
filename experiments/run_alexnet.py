#coding=utf-8
'''
Created on 2016年10月7日

@author: dengdan
'''
import time
import numpy as np
import theano.tensor as T
import theano

import alexnet.AlexNet
import data.imagenet2012_data
import nnet.layer
import util.statistic
import util.io

input = T.tensor4()
label = T.ivector()
lr = T.dscalar()
net = alexnet.AlexNet.AlexNet(input, label)

print 'building theano functions...'
t1 = time.time()
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
debug_items = [
                  ["input", net.input],
                  ["conv1 output",net.conv1], 
                  ["conv3_1",net.conv3_1], 
                  ["conv3_2", net.conv3_2], 
                  ["conv4_1",  net.conv4_1],
                  ["conv4_2",net.conv4_2], 
                  ["conv5_1",net.conv5_1], 
                  ["conv5_2",  net.conv5_2], 
                  ["fc1_1", net.fc1_1], 
                  ["fc1_2", net.fc1_2], 
                  ["fc2_1",  net.fc2_1], 
                  ["fc2_2",  net.fc2_2], 
                  ["fc2_output", net.fc2_output], 
                  ["net.predicted",  net.predicted],
                  ["net.conv1.W",net.conv1.W]
          ]
debug_output = []
for idx, result in enumerate(debug_items):
    if isinstance(result[-1], nnet.layer.Layer):
        debug_output.append(result[-1].lin_output)
    else:
        debug_output.append(result[-1])

debug = theano.function(
                        inputs = [input],
                        outputs = debug_output 
                        )

t2 = time.time()
print 'building finished. %f seconds used' % (t2 - t1)

print "There are %d parameters in AlexNet"%( get_param_count())
epochs = 20
learning_rate = 0.00001
training_losses = []
training_accuracies = []
val_losses = []
val_accuracies = []
epoch_time = []
dump_path = './model/'
batch_size = 100

print "creating data iterator..."
imagenet2012_data_dir = '/home/dengdan/dataset/imagenet_2012_mxnet/'
train_iter =  data.imagenet2012_data.get_imagenet_2012_train_data(data_path = imagenet2012_data_dir,batch_size = batch_size)
val_iter = data.imagenet2012_data.get_imagenet_2012_val_data(data_path = imagenet2012_data_dir ,batch_size = batch_size)
# train_iter = val_iter
print "creating data iterator OK."

util.io.mkdir(dump_path)

for epoch in xrange(epochs):
    iteration = 0
    train_iter.reset()
    t1 = time.time()
    for batch in train_iter:
        data_X = batch.data[0].asnumpy()
        data_y = np.asarray(batch.label[0].asnumpy(), dtype = np.int32)
        t2 = time.time()
        io_time = t2 - t1
#         debug_info = debug(data_X)
#         for idx, info in enumerate(debug_items):
#             title = "Epoch %d, Iteration %d, %s"%(epoch, iteration, info[0])
#             to_be_shown = debug_info[idx]
#             print "%s: D = %f, E = %f"%(title, util.statistic.D(to_be_shown), util.statistic.E(to_be_shown))
        training_loss, training_accuracy = train_alexnet(data_X, data_y, learning_rate)
        t1 = time.time()
        training_time = t1 - t2
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        
        print "Epoch %d, Iteration %d: loss = %f, accuracy = %f, training time = %f seconds, io time = %f seconds" % (epoch, iteration, training_loss, training_accuracy, training_time, io_time)
        iteration += 1
    
    # validation
    val_loss = []
    val_accuracy = []
    print 'validating model...'
    for batch in val_iter:
        data_X = batch.data[0].asnumpy()
        data_y = np.asarray(batch.label[0].asnumpy(), dtype = np.int32)
        val_loss_batch, val_accuracy_batch = train_alexnet(data_X, data_y, learning_rate)
        val_loss.append(val_loss_batch)
        val_accuracy.append(val_accuracy_batch)
    val_iter.reset()
    val_losses.append(np.mean(val_loss))
    val_accuracies.append(np.mean(val_accuracy))
    print "Epoch %d, validation: loss = %f, accuracy = %f"%(epoch, val_losses[-1], val_accuracies[-1])
    
    # dump data and model
    dump_model = 'Alexnet_Epoch%d.model'%(epoch)
    dump_data = 'Alexnet_Epoch%d.data'%(epoch)
    t2 = time.time()
    print 'dumping model and data...'
    util.io.dump(util.io.join_path(dump_path, dump_model), net)
    util.io.dump(util.io.join_path(dump_path, dump_data), {
                                                            "training loss": training_losses,
                                                            "training accuracy": training_accuracies,
                                                            'validation loss': val_losses,
                                                            'validation accuracies': val_accuracies
                                                            })
    t1 = time.time()
    print 'dumping finished, time used = %f seconds...'%(t1 - t2)
