# encoding = utf-8
def int_array_to_str(arr):
    """turn an int array to a str"""
    return "".join(map(chr, arr))

def join(arr, splitter=','):
    temp = []
    for e in arr:
        if not is_str(e):
            e = str(e)
        temp.append(e)
        temp.append(splitter)
    temp.pop()
    return "".join(temp)

def join_(arr, splitter=","):
    temp = []
    for e in arr:
        if is_str(e):
            e = e.encode('utf-8')
        else:
            e = str(e)
        temp.append(e)
    return splitter.join(temp)
 
def is_str(s):
    try:
        return type(s) == str or type(s) == unicode
    except:
        return isinstance(s, str)

def to_lowercase(s):
    return str.lower(s)

def to_uppercase(s):
    return str.upper(s)
    
def ends_with(s, suffix, ignore_case = False):
    """
    suffix: str, list, or tuple
    """
    if is_str(suffix):
        suffix = [suffix]
    suffix = list(suffix)
    if ignore_case:
        for idx, suf in enumerate(suffix):
            suffix[idx] = to_lowercase(suf)    
        s = to_lowercase(s)
    suffix = tuple(suffix)
    return s.endswith(suffix)

def starts_with(s, prefix, ignore_case = False):
    """
    prefix: str, list, or tuple
    """
    if is_str(prefix):
        prefix = [prefix]
    prefix = list(prefix)
    if ignore_case:
        for idx, pre in enumerate(prefix):
            prefix[idx] = to_lowercase(pre)    
        s = to_lowercase(s)
    prefix = tuple(prefix)
    return s.startswith(prefix)


def contains(s, target, ignore_case = False):
    if ignore_case:
        s = to_lowercase(s)
        target = to_lowercase(target)
    return s.find(target) >= 0

def index_of(s, target):
    return s.find(target)

def replace_all(s, old, new, reg = False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s
    
def remove_all(s, sub):
    return replace_all(s, sub, '')
    
def split(s, splitter, reg = False):
    if not reg:
        return s.split(splitter)
    import re
    return re.split(splitter, s)   
    
def remove_invisible(s):
    s = replace_all(s, ' ', '')
    s = replace_all(s, '\n', '')
    s = replace_all(s, '\t', '')
    s = replace_all(s, '\r', '')
    s = replace_all(s, '\xef\xbb\xbf', '')
    return s

def trim(s):
    return s.strip()
    
def find_all(s, pattern):
    import re
    return re.findall(pattern, s)

def is_none_or_empty(s):
    if s is None:
        return True
    return len(s)==0; 

def to_json(obj, indent = 4):
    import ujson
    return ujson.dumps(obj, indent = indent)

def has_visible_chars(s):
    return len(remove_invisible(s)) > 0



