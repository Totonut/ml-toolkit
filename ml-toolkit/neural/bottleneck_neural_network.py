# -*- coding: utf-8 -*-

from activation import Sigmoid
from distance import distest
from neural_network import NeuralNetwork

class BottleneckNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, activation=Sigmoid, loss=distest(), seed=None):
        super(BottleneckNeuralNetwork, self).__init__(layers, activation, loss, seed)
        
    def predict(self, inputs, bnl=-1):
        results = inputs
        for i in range(bnl if bnl > 0 else len(self.layers)):
            results = self.layers[i].predict(results)
        return results