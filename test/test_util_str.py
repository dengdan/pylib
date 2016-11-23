#encoding = utf-8

from common_import import *

from util.str import *

@util.dec.print_test
def test_to_lowercase():
    np.testing.assert_string_equal(to_lowercase('Ss'), 'ss')
    

@util.dec.print_test
def test_endswith():
    np.testing.assert_(endswith('hello.ss', 'ss'))
    np.testing.assert_(endswith('hello.ss', '.SS', ignore_case = True))
    
    np.testing.assert_(endswith('hello.ss', ['ss', 'SS']))
    
@util.dec.print_test
def test_is_str():
    np.testing.assert_(is_str(''))
    np.testing.assert_(not is_str([]))
    np.testing.assert_(not is_str(0))

@util.dec.print_test
def test_contains():
    s = 'This is China'
    target = 'this'
    np.testing.assert_(not contains(s, target, ignore_case = False))
    np.testing.assert_(contains(s, target, ignore_case = True))


def test_replace_all():
    s = 'a  \t b\t  c'
    r = replace_all(s, ' ', '')
    r = replace_all(r, '\t', '')    
    np.testing.assert_equal(r, 'abc')

#test_to_lowercase()
#test_endswith()
#test_is_str()
#test_contains()
test_replace_all()
