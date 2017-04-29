#import tensorflow as tf

def summary_image(image, bboxes = None, name='image', fmt = "bhwc"):
    """Add image with bounding boxes to summary.
    """
    if fmt == "bhw":
        image = tf.cast(image, tf.float32);
        image = tf.transpose(image, [1, 2, 0]);        
        image = tf.expand_dims(image, 0)
    
    if bboxes is not None:
        if len(bboxes.shape) == 2:
            bboxes = tf.expand_dims(bboxes, 0);
        image = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image)
