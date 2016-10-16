#coding=utf-8
'''
Created on 2016年10月13日

@author: dengdan
'''
import util.log
from util.log import info
import util.io
import logging
log_file = '~/temp/log.txt'
util.log.init_logger(log_file)
info('hello0000000.')
print util.io.cat(log_file)