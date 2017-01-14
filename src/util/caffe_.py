# encoding=utf-8


def get_loss(solver, name = 'loss'):
    loss = solver.net.blobs[name].data
    return loss
