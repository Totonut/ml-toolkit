# -*- coding: utf-8 -*-

import numpy as np

class Activation():
    def activate(self, x):
        raise NotImplementedError
    
    def derivate(self, x):
        raise NotImplementedError

class Identity(Activation):
    def activate(self, x):
        return x
    
    def derivate(self, x):
        return np.ones(x.shape[0])

class ReLu(Activation):
    def activate(self, x):
        return np.where(x > 0, x, 0)
    
    def derivate(self, x):
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    def activate(self, x):
        return 1/(1 + np.exp(-x))
    
    def derivate(self, x):
        sigmoid = self.activate(x)
        return sigmoid * (1 - sigmoid)

class TanH(Activation):
    def activate(self, x):
        return np.tanh(x)
    
    def derivate(self, x):
        return 1.0 - np.tanh(x)**2

class Softmax(Activation):
    def activate(self, x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
    
    def derivate(self, x):
        soft = self.activate(x)
        return soft * (1 - soft)