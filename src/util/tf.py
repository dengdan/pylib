try:
    import tensorflow as tf
    slim = tf.contrib.slim
except:
    print "tensorflow is not installed, util.tf can not be used."
    
def is_gpu_available(cuda_only=True):
  """
  code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
  Returns whether TensorFlow can access a GPU.
  Args:
    cuda_only: limit the search to CUDA gpus.
  Returns:
    True iff a gpu device of the requested kind is available.
  """
  from tensorflow.python.client import device_lib as _device_lib

  if cuda_only:
    return any((x.device_type == 'GPU')
               for x in _device_lib.list_local_devices())
  else:
    return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
               for x in _device_lib.list_local_devices())



def get_available_gpus(num_gpus = None):
    """
    Modified on http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    However, the original code will occupy all available gpu memory.  
    The modified code need a parameter: num_gpus. It does nothing but return the device handler name
    It will work well on single-maching-training, but I don't know whether it will work well on a cluster.
    """
    if num_gpus == None:
        from tensorflow.python.client import device_lib as _device_lib
        local_device_protos = _device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    else:
        return ['/gpu:%d'%(idx) for idx in xrange(num_gpus)]

def get_latest_ckpt(path):
# tf.train.latest_checkpoint
    import util
    path = util.io.get_absolute_path(path)
    if util.io.is_dir(path):
        ckpt = tf.train.get_checkpoint_state(path)
        ckpt_path = ckpt.model_checkpoint_path
    else:
        ckpt_path = path; 
    return ckpt_path
    
def get_all_ckpts(path):
    ckpt = tf.train.get_checkpoint_state(path)
    all_ckpts = ckpt.all_model_checkpoint_paths
    ckpts = [str(c) for c in all_ckpts]
    return ckpts

def get_iter(ckpt):
    import util
    iter_ = int(util.str.find_all(ckpt, '.ckpt-\d+')[0].split('-')[-1])
    return iter_

def get_init_fn(flags):
    """
    code from github/SSD-tensorflow/tf_utils.py
    Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if flags.checkpoint_path is None:
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(flags.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % flags.train_dir)
        return None

    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in flags.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if flags.checkpoint_model_scope is not None:
        variables_to_restore = {var.op.name.replace(flags.model_name, flags.checkpoint_model_scope): var for var in variables_to_restore}

    checkpoint_path = get_latest_ckpt(flags.checkpoint_path)
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags = None):
    """code from github/SSD-tensorflow/tf_utils.py
    Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if flags is None or flags.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train
