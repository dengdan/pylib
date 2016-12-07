import log
import dtype
import plt
import np
import img
import dec
import rand
import mod
import proc
import test
import neighbour as nb
import mask
import str_ as str
import io as sys_io
import io_ as io
import feature
import thread_ as thread

# log.init_logger('~/temp/log/log_' + get_date_str() + '.log')

def exit(code = 0):
    import sys
    sys.exit(0)
    
is_main = mod.is_main
init_logger = log.init_logger
