def cross_entropy(p, target):
    """
    shape = (n_examples, ...)
    """
    p = T.flatten(p, ndim = 2)
    target = T.flatten(target, ndim = 2)
    input = target * T.log(p) + (1 - target) * T.log(1 - p)
    return T.mean(T.sum(input =  - input, axis = 1))
