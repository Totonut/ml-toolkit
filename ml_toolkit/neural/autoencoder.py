import numpy as np
from .activation import TanH
from .neural_network import NeuralNetwork

class AutoEncoder(NeuralNetwork):
    def __init__(self, layers, activation=TanH, loss=np.subtract, seed=None):
        if (type(layers[0]) is int):
            layers += layers[-2::-1]
        super(AutoEncoder, self).__init__(layers, activation, loss, seed)

    def encode(self, inputs):
        results = inputs
        nb_layers   = len(self.layers)
        for i in range(nb_layers / 2):
            results = self.layers[i].predict(results)
        return results

    def decode(self, inputs):
        results     = inputs
        nb_layers   = len(self.layers)
        for i in range(nb_layers / 2, nb_layers):
            results = self.layers[i].predict(results)
        return results

    def train(self, inputs, epochs=2500, learning_rate=0.1, momentum=0.1):
        super(AutoEncoder, self).train(inputs, inputs, epochs, learning_rate, momentum)
    
    def train_with_noise(self, inputs, seed=None, freq=10, epochs=2500, learning_rate=0.1, momentum=0.1):
        inputs  = np.array(inputs)        
        outs    = inputs
        size    = len(inputs[0])
        rand    = np.random.RandomState(seed)
        for _ in range(epochs):
            inps = [np.where(rand.randint(freq, size=size) != 0, x, 0) for x in inputs]
            super(AutoEncoder, self).train(inps, outs, 1, learning_rate, momentum)
