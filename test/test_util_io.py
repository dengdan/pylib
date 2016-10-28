# encoding = utf-8
from common_import import *
from util.io import *

@util.dec.print_test
def test_ls():
    np.testing.assert_equal(ls('.', suffix = '.py'), ls('.', suffix = '.PY'))


test_ls()
