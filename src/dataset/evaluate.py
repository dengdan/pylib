import numpy as np

import util

def iou(c, gt, I):
    """
    c: candidate proposal, a list of points
    gt: ground truth, 8 points in clockwise
    """
    gt_mask = util.img.black(I)
    util.img.fill_bbox(gt_mask, gt, color = 1)
    
    c_mask = util.img.black(I)
    util.img.render_points(c_mask, c, color = 1)
    
    union_mask = ((gt_mask + c_mask) >=1) * 1
    intersect_mask = (gt_mask * c_mask >= 1) * 1
        
    #util.plt.show_images(titles = ['Gt%d'%(np.sum(gt_mask)),'Proposal%d'%(np.sum(c_mask)), 'Union%d'%(np.sum(union_mask)), 'Intersection%d'%(np.sum(intersect_mask))], images = [gt_mask,c_mask, union_mask, intersect_mask])
    return np.sum(intersect_mask) * 1.0 / np.sum(union_mask)



@util.dec.print_test
def _test_iou():
    shape = (100, 100)
    c_mask = util.img.black(shape)
    util.img.rectangle(c_mask, (10, 30), (60, 60), color = 1, border_width = -1)
    c = util.mask.find_white_components(c_mask)[0]
    
    gt = [(30, 10), (90, 10), (90, 45), (30, 45)]
    
    a1 = util.img.rect_area((10, 30), (60, 60))
    a2 = util.img.rect_area((30, 10), (90, 45))   
    a_inter = util.img.rect_area((30, 30), (60, 45))
    a_u = a1 + a2 - a_inter
    result = iou(c, gt, c_mask)
    util.test.assert_almost_equal(result, a_inter * 1.0 / a_u)

if util.mod.is_main(__name__):
    util.init_logger()
    _test_iou()
