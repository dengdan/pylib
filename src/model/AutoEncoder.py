#encoding = utf-8

from nnet.layer import *
from nnet.fn import cross_entropy
from nnet.fn import sigmoid
import util.t
import model.Model as Model

class AutoEncoder(Model):
    """
    An AutoEncoder consists of three layers: input layer, hidden layer, and reconstruct layer.
    The hidden layer is a fully  connected layer: hfc, with params = [W, b], and the output of the hidden layer is: hfc.activation(input * W + b)
    The reconstruction layer is also a fully connected layer: rfc, with its own W or rfc.W = hfc.W.T
    """
    def __init__(self, n_visible, n_hidden, noise_level = None, share_weight = True, loss_fn = cross_entropy, name = 'AutoEncoder'):
        Model.__init__(self, name)
        self.input = T.matrix()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.noise_level = noise_level
        self.share_weight = share_weight
        
        if noise_level is not None:
            self.input = util.t.add_noise(self.input, noise_level = noise_level)
            
        hidden_layer = FullyConnectedLayer(input = self.input, n_in = n_visible, n_out = n_hidden, activation = sigmoid, name='HiddenLayer')
        
        self.hidden_layer = hidden_layer
        
        self.output = self.hidden_layer.output
        
        reconstruct_W = None
        if share_weight:
            reconstruct_W = self.hidden_layer.W.T
        reconstruct_layer = FullyConnectedLayer(input = self.output, n_in = n_hidden, n_out = n_visible, activation = None, W = reconstruct_W)
        self.reconstruct_layer = reconstruct_layer
        self.reconstructed = self.reconstruct_layer.output
        self.loss = loss_fn(self.reconstructed, self.input)
        
        self.layers = [hidden_layer, reconstruct_layer]
        self.touch_params()
