#coding=utf-8
'''
Created on 2016年10月13日

@author: dengdan
'''

import model.Model as Model
import theano.tensor as T
import nnet.layer
class LeNet(Model):
    """input_size = (3, 224, 224)"""
    def __init__(self, name = "LeNet"):
        
        Model.__init__(self, name)
        self.conv1_kernel_shape = (12, 3, 5, 5)
        self.conv1_padding = (2, 2)
        self.conv1_stride = (2, 2)
        # conv1 输出为12* 112 * 112
        
        self.conv2_kernel_shape = (12, 12, 5, 5)
        self.conv2_padding = (2, 2)
        self.conv2_stride = (2, 2)

        
        self.fc1_input_units = 12 * 56 * 56        
        self.fc1_hidden_units = 1001
        
        self.conv1 = nnet.layer.ConvolutionLayer(input = self.input, filter_shape =  self.conv1_kernel_shape, stride = self.conv1_stride, padding = self.conv1_padding, name = "conv1")
        self.conv2 = nnet.layer.ConvolutionLayer(input = self.conv1, filter_shape= self.conv2_kernel_shape, stride= self.conv2_stride, padding= self.conv2_padding, name = 'conv2')
        self.conv2_flatten = T.flatten(self.conv2.output, outdim = 2)
        self.output_layer = nnet.layer.SoftmaxOutputLayerWithLoss(input = self.conv2_flatten, n_in = self.fc1_input_units, n_out = self.fc1_hidden_units, label = self.label, name = 'fc1')
        
        self.layers = [self.conv1, self.conv2, self.output_layer]
        self.touch_params()
            
        self.loss = self.output_layer.loss
        self.accuracy = self.output_layer.accuracy
        self.predicted = self.output_layer.predicted    
        
        
        
        
        
