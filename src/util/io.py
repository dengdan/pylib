#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan

Tool  functions for file system operation and I/O. 
In the style of linux shell commands
'''
import os
import cPickle as pkl

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created. 
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def pwd():
    return os.getcwd()

def dump(path, obj):
    with open(path, 'w') as f:
        pkl.dump(obj, f)
        
def join_path(a, *p):
    return os.path.join(a, *p)

def get_absolute_path(p):
    return os.path.abspath(p)

def cd(p):
    os.chdir(p)

def ls(dir = '.'):
    return os.listdir(dir)

def exists(path):
    return os.path.exists(path)
import util.mod
if util.mod.is_main(__name__):
    print (ls())
    
