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
import logging

import util.str

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created. 
    """
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
        
def make_parent_dir(path):
    """make the parent directories for a file."""
    parent_dir = get_dir(path)
    mkdir(parent_dir)
    
    
def pwd():
    return os.getcwd()

def dump(path, obj):
    path = get_absolute_path(path)
    parent_path = get_dir(path)
    mkdir(parent_path)
    with open(path, 'w') as f:
        logging.info('dumping file:' + path);
        pkl.dump(obj, f)

def load(path):
    path = get_absolute_path(path)
    with open(path, 'r') as f:
        data = pkl.load(f)
    return data

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
    
def ls(path = '.', suffix = None):
    """
    list files in a directory.
    return file names in a list
    """
    path = get_absolute_path(path)
    files = os.listdir(path)

    if suffix is None:       
        return files
        
    filtered = []
    for f in files:
        if util.str.endswith(f, suffix, ignore_case = True):
            filtered.append(f)
    
    return filtered
    
def cat(p):
    cmd = 'cat ' + p
    return commands.getoutput(cmd)

def exists(path):
    return os.path.exists(path)


import util.mod
if util.mod.is_main(__name__):
    print get_absolute_path("~/dataset")
