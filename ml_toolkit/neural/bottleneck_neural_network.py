import numpy as np
from .activation import TanH
from .neural_network import NeuralNetwork

class BottleneckNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, activation=TanH, loss=np.subtract, seed=None):
        super(BottleneckNeuralNetwork, self).__init__(layers, activation, loss, seed)
        
    def encode(self, inputs, end):
        results = inputs
        for i in range(end):
            results = self.layers[i].predict(results)
        return results

    def decode(self, inputs, start):
        results = inputs
        for i in range(start, len(self.layers)):
            results = self.layers[i].predict(results)
        return results
