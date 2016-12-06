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

import util

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
    '''
    return the directory it belongs to
    '''
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

def read_lines(p):
    """return the text in a file in lines as a list """
    p = get_absolute_path(p)
    f = open(p,'r')
    return f.readlines()
    
def write_lines(p, lines):
    p = get_absolute_path(p)
    with open(p, 'w') as f:
        for line in lines:
            f.write(line)
            

def cat(p):
    """return the text in a file as a whole"""
    cmd = 'cat ' + p
    return commands.getoutput(cmd)

def exists(path):
    return os.path.exists(path)

def load_mat(path):
    import scipy.io as sio
    path = get_absolute_path(path)
    return sio.loadmat(path)

def dump_mat(path, dict_obj, append = True):
    import scipy.io as sio
    path = get_absolute_path(path)
    make_parent_dir(path)
    sio.savemat(file_name = path, mdict =  dict_obj, appendmat = append)
    
def dir_mat(path):
    '''
    list the variables in mat file.
    return a list: [(name, shape, dtype), ...]
    '''
    import scipy.io as sio
    path = get_absolute_path(path)
    return sio.whosmat(path)
    
SIZE_UNIT_K = 1024
SIZE_UNIT_M = SIZE_UNIT_K ** 2
SIZE_UNIT_G = SIZE_UNIT_K ** 3
def get_file_size(path, unit = SIZE_UNIT_K):
    size = os.path.getsize(get_absolute_path(path))
    return size * 1.0 / unit
