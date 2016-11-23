# encoding = utf-8

def is_str(s):
    return type(s) == str

def to_lowercase(s):
    return str.lower(s)
    
def endswith(s, suffix, ignore_case = False):
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

def contains(s, target, ignore_case = False):
    if ignore_case:
        s = to_lowercase(s)
        target = to_lowercase(target)
    return s.find(target) >= 0
    
def replace_all(s, old, new):
    return s.replace(old, new)
