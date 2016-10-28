# encoding=utf-8
from common_import import *

@util.dec.print_test
def test_normal():
    shape = (500, 500)
    mu = 10
    sigma_square = 20
    a = util.rand.normal(shape = shape, mu = mu, sigma_square = sigma_square)
    E = util.statistic.E(a)
    D = util.statistic.D(a)
    np.testing.assert_almost_equal(E, mu, 0)
    np.testing.assert_almost_equal(D, sigma_square, 0)

@util.dec.print_test
def test_randint():
#    logging.info('generating random int:%d'%(util.rand.randint()))
    print util.rand.randint()
    print util.rand.randint(10)
    print util.rand.randint(shape = (2, 3))
test_randint()
test_normal()
