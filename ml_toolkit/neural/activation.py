import numpy as np
from abc import *

class Activation():
    __metaclass__ = ABCMeta
    @staticmethod
    @abstractmethod
    def activate(x):
        pass

    @staticmethod
    @abstractmethod
    def derivate(x):
        pass

class Identity(Activation):
    @staticmethod
    def activate(x):
        return x
    
    @staticmethod
    def derivate(x):
        return np.ones(x.shape[0])

class ReLu(Activation):
    @staticmethod
    def activate(x):
        return np.where(x > 0, x, 0)
    
    @staticmethod
    def derivate(x):
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    @staticmethod
    def activate(x):
        return 1/(1 + np.exp(-x))
    
    @staticmethod
    def derivate(x):
        sigmoid = Sigmoid.activate(x)
        return sigmoid * (1 - sigmoid)

class TanH(Activation):
    @staticmethod
    def activate(x):
        return np.tanh(x)
    
    @staticmethod
    def derivate(x):
        return 1.0 - np.tanh(x)**2

class Softmax(Activation):
    @staticmethod
    def activate(x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
    
    @staticmethod
    def derivate(x):
        soft = Softmax.activate(x)
        return soft * (1 - soft)
