#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan

Tool  functions for file system operation and I/O. 
In the style of linux shell commands
'''
import os
import cPickle as pkl
import commands

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created. 
    """
    path = get_absolute_path(path)
    os.makedirs(path)

def pwd():
    return os.getcwd()

def dump(path, obj):
    parent_path = get_absolute_path(path)
    mkdir(parent_path)
    with open(path, 'w') as f:
        pkl.dump(obj, f)
        
def join_path(a, *p):
    return os.path.join(a, *p)

def get_dir(path):
    path = get_absolute_path(path)
    return os.path.split(path)[0]

def get_filename(path):
    return os.path.split(path)[1]

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def cd(p):
    p = get_absolute_path(p)
    os.chdir(p)
    
def ls(path = '.'):
    path = get_absolute_path(path)
    return os.listdir(path)

def cat(p):
    cmd = 'cat ' + p
    return commands.getoutput(cmd)

def exists(path):
    return os.path.exists(path)


import util.mod
if util.mod.is_main(__name__):
    print get_absolute_path("~/dataset")