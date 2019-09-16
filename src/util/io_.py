#coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan

Tool  functions for file system operation and I/O. 
In the style of linux shell commands
'''
import os
try:
    import cPickle as pkl
except:
    import pickle as pkl
import logging

import util

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created. 
    """
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
    return path

def make_parent_dir(path):
    """make the parent directories for a file."""
    parent_dir = get_dir(path)
    if not exists(parent_dir):
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

def is_dir(path):
    path = get_absolute_path(path)
    return os.path.isdir(path)

is_directory = is_dir

def is_path(path):
    path = get_absolute_path(path)
    return os.path.ispath(path)
    
def get_dir(path):
    '''
    return the directory it belongs to.
    if path is a directory itself, itself will be return 
    '''
    path = get_absolute_path(path)
    if is_dir(path):
        return path;
    return os.path.split(path)[0]

def get_parent_dir(path):
    current_dir = get_dir(path)
    return get_absolute_path(join_path(current_dir, '..'))

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
        if util.str.ends_with(f, suffix, ignore_case = True):
            filtered.append(f)
    
    return filtered

def find_files(pattern, recursive = False):
    import glob
    return glob.glob(pattern, recursive = recursive)

def read_lines(p, encoding = None):
    """return the text in a file in lines as a list """
    p = get_absolute_path(p)
    if encoding:
        import codecs
        f = codecs.open(p, 'rU', encoding)
    else:
        f = open(p,'rU')
    return f.readlines()
    
def write_lines(p, lines, append_break = True, utf8 = False):
    p = get_absolute_path(p)
    make_parent_dir(p)
    with open(p, 'w') as f:
        for line in lines:
            if append_break:
                f.write(line + '\n')
            else:
                f.write(line)

def cat(p):
    """return the text in a file as a whole"""
    cmd = 'cat ' + p
#     return commands.getoutput(cmd)
    return util.cmd.cmd(cmd)

def exists(path):
    path = get_absolute_path(path)
    return os.path.exists(path)

def not_exists(path):
    return not exists(path)

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
unit_dict = {"M" :  SIZE_UNIT_M, "K" : SIZE_UNIT_K, "G" : SIZE_UNIT_G}
def get_file_size(path, unit = SIZE_UNIT_K):
    if type(unit) == str:
        import util
        try:
            unit = unit_dict[util.str.to_uppercase(unit)[0]]
        except:
            util.log.error("Unkown filesize unit: " + unit)
    size = os.path.getsize(get_absolute_path(path))
    return size * 1.0 / unit
    
    
def create_h5(path):
    import h5py
    path = get_absolute_path(path)
    make_parent_dir(path)
    return h5py.File(path, 'w');

def open_h5(path, mode = 'r'):
    import h5py
    path = get_absolute_path(path)
    return h5py.File(path, mode);
    
def read_h5(h5, key):
    return h5[key][:]
def read_h5_attrs(h5, key, attrs):
    return h5[key].attrs[attrs]
    
def copy(src, dest):
    util.io.make_parent_dir(dest)
    import shutil
    if is_dir(src):
        fn = shutil.copytree
    else:
        fn = shutil.copy
    fn(get_absolute_path(src), get_absolute_path(dest))
    
cp = copy

def remove(p):
    import os
    os.remove(get_absolute_path(p))
rm = remove

def search(pattern, path, file_only = True, recursive = False):
    """
    Search files whose name matches the give pattern. The search scope
    is the directory and sub-directories of 'path'. 
    """
    path = get_absolute_path(path)
    pattern_here = util.io.join_path(path, pattern)
    targets = []
    
    # find matchings in current directory
    candidates = find_files(pattern_here, recursive = recursive)
    for can in candidates:
        if util.io.is_dir(can) and file_only:
            continue
        else:
            targets.append(can)
            
    # find matching in sub-dirs
    files = ls(path)
    for f in files:
        fpath = util.io.join_path(path, f)
        if is_dir(fpath):
            targets_in_sub_dir = search(pattern, fpath, file_only, recursive = recursive)
            targets.extend(targets_in_sub_dir)
    return targets

def dump_json(path, data):
    try:
        import ujson as json
    except:
        import json
    path = get_absolute_path(path)
    make_parent_dir(path)

    with open(path, 'w') as f:
        json.dump(data, f)
    return path

def load_json(path):
    try: 
        import ujson as json
    except:
        import json
    path = get_absolute_path(path)
    with open(path, 'r')  as f:
        return json.load(f)
    
def load_pb(path, cls):
    """
    load a protobuf object from file
    path: path to the serialized object
    cls: the class of the object
    """
    from google.protobuf import text_format
    obj = cls()
    with open(get_absolute_path(path)) as f:
        text_format.Merge(f.read(), obj)
    return obj

    
