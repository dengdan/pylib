import cv2
import logging
def wait_key(target = None, callback = None):
    key = cv2.waitKey()& 0xFF
    if target == None:
        return key
    if type(target) == str:
        target = ord(target)
    while key != target:
        key = cv2.waitKey()& 0xFF

    logging.debug('Key Pression caught:%s'%(target))
    
    if callback is not None:
        callback()

