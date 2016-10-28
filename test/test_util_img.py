import util.img
import numpy as np
def test_ds_size():
    image_size = (15, 15)
    kernel_size = (2, 2)
    stride = (1, 1)
    np.testing.assert_equal(util.img.ds_size(image_size, kernel_size, stride), (14, 14))
    
    kernel_size = (2, 2)
    stride = (3, 3)
    np.testing.assert_equal(util.img.ds_size(image_size, kernel_size, stride), (5, 5))
    
    kernel_size = (3, 3)
    stride = (2, 2)
    np.testing.assert_equal(util.img.ds_size(image_size, kernel_size, stride), (7, 7))
    
    kernel_size = (3, 2)
    stride = (3, 2)
    np.testing.assert_equal(util.img.ds_size(image_size, kernel_size, stride), (5, 7))
    

test_ds_size()
