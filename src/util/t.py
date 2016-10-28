#encoding=utf-8
"""
for theano shortcuts
"""
import theano
import theano.tensor as T
import util.rand

trng = T.shared_randomstreams.RandomStreams(util.rand.randint())
scan_until = theano.scan_module.until

def add_noise(input, noise_level):
    noise = trng.binomial(size = input.shape, n = 1, p = 1 - noise_level)
    return noise * input

