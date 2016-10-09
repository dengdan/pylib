#coding=utf-8
'''
Created on 2016年10月6日

@author: dengdan
'''
import nnet.layer as layer
import theano.tensor as T
class AlexNet(object):
    def __init__(self, input, label):
        self.input = input
        
        self.conv1_filter_shape = (96, 3, 11, 11)
        self.conv2_filter_shape = (128,48,5,5)
        self.conv3_filter_shape = (192,256,3,3)
        self.conv4_filter_shape = (192, 192, 3,3)
        self.conv5_filter_shape = (128, 192, 3, 3)
        
        self.conv1_stride = (4, 4)
        
        # alexnet的pooling使用相同的参数.
        self.pool_filter_shape = (3, 3)
        self.pool_stride = (2, 2)
        
        self.fc_hidden_units = 2048
        self.output_units = 1000
        
        
        # padding: conv1的输出为55*55, stride = 4, kernel_size = 11, 需要input_size = 54 * 4 + 11 - 1 = 226, 四条边各padding 2
        self.conv1_padding = 2
        conv1 = layer.ConvolutionLayer(input = input, filter_shape = self.conv1_filter_shape, stride = self.conv1_stride, padding = self.conv1_padding)
        
        # pool1 的输出为27*27, stride = 2, kernel_size = 3, input_size = 26 * 2 + 3 - 1 = 54, 不需要padding, 需要忽略最后一列/行
        pool1 = layer.MaxPooling(input = conv1, filter_shape = self.pool_filter_shape, stride = self.pool_stride, ignore_border = True)
        
        # 第二层开始分开了.
        pool1_columns = T.split(x = pool1.output, splits_size = [48, 48], n_splits = 2, axis = 1)
        
        # pool2的输出为13*13, 反推conv2的输出最多为 12 * 2 + 3 - 1 + 1 = 27, 选择保持不变.  stride = 2肯定是不行的, 所以只能为1. 
        # 那么最后一个输出对应的input 位置为: 26 *  1 + 5 - 1 = 30, padding取2
        self.conv2_stride = (1, 1)
        self.conv2_padding = 2 # or (2, 2)
        def get_conv2_pool2_column(idx):
                conv2_column = layer.ConvolutionLayer(input = pool1_columns[idx-1], filter_shape = self.conv2_filter_shape, stride = self.conv2_stride, 
                                                      padding = self.conv2_padding, name = 'conv2_%d'%(idx))
                pool2_column = layer.MaxPooling(input = conv2_column, filter_shape = self.pool_filter_shape, stride = self.pool_stride)
                return [conv2_column, pool2_column]
        conv2_1, pool2_1 = get_conv2_pool2_column(1)
        conv2_2, pool2_2 = get_conv2_pool2_column(2)
        
        # 组成256个feature map
        pool2_stacked = T.concatenate([pool2_1.output, pool2_2.output], axis = 1)
        
        # conv3到conv5保持形状不变, 最后一个conv对应 12 * 1 + 3 - 1 = 14, 16 > input_size >=15, 取input_size = 15, padding = 1
        self.conv3_conv4_conv5_stride = (1, 1)        
        self.conv3_conv4_conv5_padding = 1
        def get_conv3_conv4_conv5_fc1_fc2_column(idx):
            conv3_column = layer.ConvolutionLayer(input = pool2_stacked ,                filter_shape = self.conv3_filter_shape, stride = self.conv3_conv4_conv5_stride, padding = self.conv3_conv4_conv5_padding, name = 'conv3_%d'%(idx))
            conv4_column = layer.ConvolutionLayer(input = conv3_column , filter_shape = self.conv4_filter_shape, stride = self.conv3_conv4_conv5_stride, padding = self.conv3_conv4_conv5_padding, name = 'conv4_%d'%(idx))
            conv5_column = layer.ConvolutionLayer(input = conv4_column , filter_shape = self.conv5_filter_shape, stride = self.conv3_conv4_conv5_stride, padding = self.conv3_conv4_conv5_padding, name = 'conv5_%d'%(idx))
            conv5_column_flaten = T.flatten(conv5_column.output,outdim = 2)
            fc1_column = layer.FullyConnectedLayer(input = conv5_column_flaten, n_in = 13*13*128, n_out = self.fc_hidden_units, activation = T.nnet.relu)
            fc2_column = layer.FullyConnectedLayer(input = fc1_column, n_in = self.fc_hidden_units, n_out = self.fc_hidden_units, activation = T.nnet.relu)
            return [conv3_column, conv4_column, conv5_column, fc1_column, fc2_column]
        
        conv3_1, conv4_1, conv5_1, fc1_1, fc2_1 = get_conv3_conv4_conv5_fc1_fc2_column(1)
        conv3_2, conv4_2, conv5_2, fc1_2, fc2_2 = get_conv3_conv4_conv5_fc1_fc2_column(2)
        
        fc2_output = T.concatenate([fc2_1.output, fc2_2.output], axis = 1)
        output_layer = layer.SoftmaxOutputLayerWithLoss(input = fc2_output, n_in = 2*self.fc_hidden_units, n_out = self.output_units, label = label, name = 'SoftmaxOutputLayerWithLoss')
        self.loss = output_layer.loss
        self.accuracy = output_layer.accuracy
        self.predicted = output_layer.predicted
        
        self.layers = [conv1, pool1,conv2_1, pool2_1,conv2_2, pool2_2, conv3_1, conv4_1, conv5_1, fc1_1, fc2_1, conv3_2, conv4_2, conv5_2, fc1_2, fc2_2 , output_layer]
        self.params = []
        for l in self.layers:
            self.params.extend(l.params)
        
        
    def get_updates(self, lr):    
        updates = [(p, p - lr * T.grad(self.loss, p)) for p in self.params]
        return updates
        
        
    