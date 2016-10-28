#encoding=utf-8
import logging
import time
def print_calling(fn):
    def wrapper(*args1, ** args2):
        s = "calling function %s"%(fn.__name__)
        logging.info(s)
        start = time.time()
        ret = fn(*args1, **args2)
        end = time.time()
        s = " time used = %f seconds"%(end - start)
        logging.info(s)
        return ret
    return wrapper


def print_test(fn):
    def wrapper(*args1, ** args2):
        s = "running test: %s..."%(fn.__name__)
        logging.info(s)
        ret = fn(*args1, **args2)
        s = "running test: %s...succeed"%(fn.__name__)
        logging.info(s)
        return ret
    return wrapper
    

