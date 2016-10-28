import numpy as np

import util.np
import util.rand
import util.mod
import util.dec

@util.dec.print_test
def test_flatten():
    a = util.rand.rand(3, 5, 2)
    b = util.np.flatten(a, 2)
    b = np.reshape(b, a.shape)
    assert util.np.sum_all(b == a) == np.prod(a.shape)


@util.dec.print_test
def test_arcsin():
    pi = np.pi
    
    # max/min, 1st phase, 2nd and 3rd phase, 4th phase
    sins = [
        [1, -1],
        [1/np.sqrt(2), np.sqrt(3)/2],
        [0.5, -0.5],
        [-1/np.sqrt(2), -np.sqrt(3)/2]
    ]
    
    xs = [
        [0, 0],
        [1, 1], 
        [-1, -1],
        [1, 1]
    ]
    
    arcs = [
        [pi / 2, pi * 3 / 2], 
        [pi / 4, pi / 3], 
        [pi* 5 / 6, pi * 7 / 6], 
        [pi * 7 / 4, pi * 5 / 3]
    ]
    
    np.testing.assert_almost_equal(util.np.arcsin(sins = sins, xs = xs), arcs)


@util.dec.print_test
@util.dec.print_calling
def test_sin():
    # when angles are provided
    angles = util.rand.rand(3, 5)
    assert util.np.sum_all(util.np.sin(angles = angles) == np.sin(angles)) == np.prod(angles.shape)
    
    xs = util.rand.rand(4, 6)
    ys = util.rand.rand(4, 6)
    ys[0, :] = 0
    lengths = np.sqrt(xs ** 2 + ys ** 2)
    # when lengths are not provided
    sins = util.np.sin(xs = xs, ys = ys)     
    assert util.np.sum_all(sins == 0) == 6
    
    # when lengths are provided
    sins = util.np.sin(ys = ys, lengths = lengths)
    assert util.np.sum_all(sins == 0) == 6
    
    # scalar sin
    assert util.np.sin(ys = 0, lengths = 1) == 0
    np.testing.assert_almost_equal(util.np.sin(xs = 1, ys = 1) , np.sqrt(2) / 2)

if util.mod.is_main(__name__):
    import util.log
    util.log.init_logger(mode = 'a')
    test_flatten()
    test_arcsin()
    test_sin()

