# -*- coding: utf-8 -*-

import numpy as np
from activation import TanH

class Layer():
    def __init__(self, input_size, output_size, activation=TanH, seed=None):
        self.activation     = activation
        self.weights        = (np.random.RandomState(seed).random_sample(output_size * (input_size + 1)) - 0.5).reshape(output_size, input_size + 1)
        self.old_deltas     = np.zeros(output_size * (input_size + 1)).reshape(output_size, input_size + 1)

    def forward(self, inputs):
        self.inputs = np.append(inputs, 1)
        self.agregs = np.dot(self.weights, self.inputs)
        return self.activation.activate(self.agregs)

    def backward(self, errors, learning_rate=0.1, momentum=0.1):
        self_errors = self.activation.derivate(self.agregs) * errors
        self_deltas = np.outer(learning_rate * self_errors, self.inputs)
        self_old_weights, self.weights = self.weights, self.weights + self_deltas + momentum * self.old_deltas
        self.old_deltas = self_deltas
        return np.dot(self_errors, self_old_weights)[:-1]