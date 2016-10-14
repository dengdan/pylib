#coding=utf-8
'''
Created on 2016年10月12日
@author: dengdan
1. I/O策略
设置两个队列, 一个是record buffer, 一个是batch buffer. 
record buffer 由num_threads个生产线程(t_record_producer)从硬盘读取文件写入数据, 由一个消费线程(batch_producer)从中读取record生成batch放入batch buffer中
t_record_producer 读取数据: 
    while True:
        wait lock
        read 10 records from record_lst
        realease lock
        process record by reading image data from disk

t_buffer_producer 读取record_data, 生成batch
    while True:
        read record data from record_data queue
        produce batch 
        copy it to gpu
        add it to batch buffer
2. 读到最后一个batch数量不足的行为: 从record_lst 开头取
'''
import numpy as np
from Queue import Queue
from threading import Thread
import threading
import cv2
import util.io
import random
import copy
from util import dtype 
import logging

class ImageRecord(object):
    
    def __init__(self, image_path, image_label, image_id):
        self.image_path = image_path
        self.image_data = None
        self.image_label = image_label
        self.image_id = image_id
        
    def fill_image_data(self, shape):
        """根据record.image_path从硬盘读取图片"""    
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_data = cv2.resize(image, shape)
        
class Batch(object):
    def __init__(self, records, gpu = True):
        self.num_records = len(records)
        self.data = [0] * self.num_records
        self.label = [0] * self.num_records
        self.ids = [0] * self.num_records
        self.image_path = [""]*self.num_records
        for idx, record in enumerate(records):
            self.label[idx] = record.image_label
            self.data[idx] = record.image_data
            self.ids[idx] = record.image_id
            self.image_path[idx] = record.image_path
        
        self.data = np.asarray(self.data, dtype = dtype.uint8)
        #  Mini-batch of feature map  stacks for theano,   shape =  (batch size, input channels, input rows, input columns).
        self.data = np.transpose(self.data, [0, 3, 1, 2])
        self.label = np.asarray(self.label, dtype = dtype.int32)
        
            
        
    def get_data(self):
        return self.data
        
    def get_label(self):
        return self.label

    
class ImageDataIter(object):
     
    def __init__(self, root_dir = "", lst_path = "", image_shape = (224, 224), batch_size = 100, random_shuffle = True, center_cropped = True, prefetch = 5, num_threads = 5, auto_stop = False):
        self.lst_path = util.io.get_absolute_path(lst_path) # 主要是为了处理~
        self.root_dir = util.io.get_absolute_path(root_dir)
        self.image_shape = image_shape
        self.random_shuffle = random_shuffle
        self.center_cropped = center_cropped
        self.auto_stop = False
        
        logging.debug('reading lst file:' + self.lst_path)
        lst_file = open(self.lst_path, 'r')
        lines = lst_file.readlines()
        self.record_count = len(lines)
        self.batch_size = batch_size
        self.batch_per_epoch = np.ceil(self.record_count * 1.0 / self.batch_size) 
        self.batch_count = 0
        self.record_lst = [[]]*self.record_count
        
        
        logging.debug('parsing lst file...')
        for idx, line in enumerate(lines):
            line = line.strip().split("\t")
            image_id = int(line[0].strip())
            image_label = int(line[1].strip())
            image_name = line[2].strip()
            image_path = util.io.join_path(self.root_dir, image_name)
            self.record_lst[idx] = ImageRecord(image_path, image_label, image_id)
        
        #   record_lst的读锁
        self.record_pointer = 0
        self.record_pointer_lock = threading.Event()
        self.record_pointer_lock.set()
        
        # record_data与record_lst不同之处在于前者已经有图片数据, 后者没有
        self.record_data_buffer = Queue(maxsize = prefetch * self.batch_size)
        self.batch_buffer = Queue(prefetch)
        
        logging.debug('starting threads for I/O')
        for i in xrange(num_threads):
            t_record_data_producer = Thread(target= self.record_data_producer)
            t_record_data_producer.daemon = True
            t_record_data_producer.setName('t_record_data_producer_%d' %i)
            t_record_data_producer.start()
            
        t_batch_buffer_producer = Thread( target= self.batch_buffer_producer) 
        t_batch_buffer_producer.daemon = True
        t_batch_buffer_producer.setName("t_batch_buffer_producer")
        t_batch_buffer_producer.start()
        
    def record_data_producer(self):
        while True:
            records = self.read_record(10);
            for record in records:
                self.produce_record_data(record)
        
    def batch_buffer_producer(self):
        while True:
            records = [0] * self.batch_size
            for i in range(self.batch_size):
                records[i] = self.consum_record_data()
            new_batch = Batch(records)
            self.batch_buffer.put(new_batch)
                
    
    def produce_record_data(self, record):
        record.fill_image_data(self.image_shape)
        self.record_data_buffer.put(record)
    
    def consum_record_data(self):
        return self.record_data_buffer.get();
    
    def read_record(self, num_records):
        self.record_pointer_lock.wait()
        records = [0]*num_records 
        for i in range(num_records):
            if self.record_pointer > self.record_count:
                if self.random_shuffle:
                    random.shuffle(self.record_lst)
                    self.record_pointer = 0
            # 要copy后返回, 不然image data会保留在record_lst里, 造成内存消耗甚至程序崩溃.
            records[i] = copy.copy(self.record_lst[self.record_pointer])
            self.record_pointer += 1
        self.record_pointer_lock.set()
        
        return records
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.auto_stop and self.batch_count >= self.batch_per_epoch:
            raise  StopIteration
        batch = self.batch_buffer.get()
        self.batch_count += 1
        return batch
    
    def reset(self):
        self.batch_count  = 0
