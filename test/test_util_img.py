import numpy as np

import util.img
import util.plt
import util.dec
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
    
    
@util.dec.print_test
def test_blur():
    image_path = '~/dataset/ICDAR2015/Challenge4/ch4_training_images/img_150.jpg'
    image = util.img.imread(image_path)
    blurred_3 = util.img.average_blur(image, (3, 3))
    blurred_5 = util.img.average_blur(image, (5, 5))
    util.plt.show_images(images = [image, blurred_3, blurred_5], titles = ['origin', '3x3', '5x5'], bgr2rgb = True, share_axis = True)
    
@util.dec.print_test
def test_rect_perimeter():
    p = util.img.rect_perimeter((5, 5), (80, 80))
    np.testing.assert_equal(p, 300)
    
    p = util.img.rect_perimeter((0, 0), (80, 80))
    np.testing.assert_equal(p, 320)


#test_rect_perimeter()
#test_ds_size()
#test_blur()

