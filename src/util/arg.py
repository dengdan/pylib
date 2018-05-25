
def process_bool(args, names):
    from argparse import Namespace
    assert isinstance(args, Namespace)
    for name in names:
        val = getattr(args, name)
        val  = get_bool_value(val)
        setattr(args, name, val)
    return args

def get_args(parser, bools = None):
    args = parser.parse_args()
    if bools:
        process_bool(args, names = bools)
    return args

parse_args = get_args

def get_parser(description = 'Input'):
    import argparse
    parser = argparse.ArgumentParser(description= description)
    return parser

def get_bool_value(value):
    if value is None:
        return False
    
    try:
        value = int(value)
        return value != 0
    except:
        value = str(value).strip()
        if not value:
            return False
        
        if value in ['False', 'false']:
             return False
        else:
             return True
    