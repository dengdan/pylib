# encoding = utf-8
from common_import import *
from util.io import *

@util.dec.print_test
def test_ls():
    np.testing.assert_equal(ls('.', suffix = '.py'), ls('.', suffix = '.PY'))

@util.dec.print_test
def test_readlines():
    p = __file__
    lines = readlines(p)
    for l in lines:
        print l    
test_ls()
test_readlines()


