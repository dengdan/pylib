# encoding=utf-8

import util
def get_data(net, name):
    import caffe
    if isinstance(net, caffe._caffe.Solver):
        net = net.net
    return net.blobs[name].data[...]
    
def get_params(net, name = None):
    import caffe
    if isinstance(net, caffe._caffe.Solver):
        net = net.net
    params = net.params[name]
    p = []
    for param in params:
        p.append(param.data[...])
    return p
    
def draw_log(log_path, output_names, show = False, save_path = None):
    pattern = "Train net output: word_bbox_loc_loss = "
    log_path = get_absolute_path(log_path)
    f = open(log_path,'r')
    iterations = []
    outputs = {}
    for line in f.readlines():
        if util.str.contains('Iteration') and util.str.contains('lr'):
            s = line.split('Iteration')[-1]
            iter_num = util.str.find_all(s, '\d+')[0]
            iterations.append(iter_num)
            continue
        if util.str.contains(line, "Train net output #\d+"):
            s = line.split('Train net output #\d+\:')[-1]
            s = s.split('(')[0]
            output = util.str.find_all(s, '\d*\.*\d+')
            output = float(output)
            for name in output_names:
                if s.contains(name):
                    if name not in outputs:
                        outputs[name] = output
                    outputs.append(output)
                plt.plot(x, val_accuracies, 'g--', label = 'Validation Accuracy')
    colors = ['r', 'g', 'b']
    line_type = ['-', '--', '-.', '.', ':']
    for name in outputs:
        output = outputs[name]
        util.test.assert_equal(len(output) , len(iterations))
        line_style = util.plt.get_rand_line_style()
        util.plt.plot(iterations, losses, line_style, label = name)
    
    plt.legend()
    
    if save_path is not None:
        util.plt.save_image(save_path)
    if show:
        util.plt.show()
