from math import *
from abc import *
import numpy as np

from distances import *

"""
    Cost functions and derivatives to evaluate a model's parameters
    For each function:
        y are the expected outputs
        a are the outputs we got
"""

class Cost:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def loss(a, y):
        pass

    @staticmethod
    @abstractmethod
    def derivative(a, y):
        pass

class CrossEntropy(Cost):
    @staticmethod
    def loss(a, y):
        assert len(a) == len(y), "wrong arguments in cross_entropy loss function: lengths mismatch"
        return -np.sum([y * log(a) + (1 - y) * log(1 - a) for (a, y) in zip(a, y)]) / len(a)

    @staticmethod
    def derivative(a, y):
        return (a - y) / (a * (1 - a))


class Quadratic(Cost):
    @staticmethod
    def loss(a, y):
        assert len(a) == len(y), "wrong arguments in quadratic loss function: lengths mismatch"
        return np.sum([(a - y) ** 2 for (a, y) in zip(a, y)]) / (2 * len(a))

    @staticmethod
    def derivative(a, y):
        assert len(a) == len(y), "wrong arguments in quadratic loss function: lengths mismatch"
        return a - y

class Exponential(Cost):
    @staticmethod
    def loss(a, y, l=1):
        return l * exp((1 / l) * np.sum([(a - y) ** 2 for (a, y) in zip(a, y)]))

    @staticmethod
    def derivative(a, y, l=1):
        return (2 / l) * (a - y) * Exponential.loss(a, y, l)

class Hellinger(Cost):
    @staticmethod
    def loss(a, y):
        return (1 / sqrt(2)) * np.sum([(sqrt(a) - sqrt(y)) ** 2 for (a, y) in zip(a, y)])

    @staticmethod
    def derivative(a, y):
        return (np.sqrt(a) - np.sqrt(y)) / (sqrt(2) * np.sqrt(a))

class KullbackLeibler(Cost):
    """
        This evaluates the loss of information when y is used to approximate a
    """
    @staticmethod
    def loss(a, y):
        return np.sum([a * (log(a) / log(y)) for (a, y) in zip(a, y)])

    @staticmethod
    def derivative(a, y):
        return y / a

class GeneralizedKullbackLeibler(Cost):
    """
        This evaluates the loss of information when y is used to approximate a
    """
    @staticmethod
    def loss(a, y):
        return np.sum([a * (log(a) / log(y)) for (a, y) in zip(a, y)]) - np.sum(y) + np.sum(a)

    @staticmethod
    def derivative(a, y):
        return (y + a) / a
        

class ItakuraSaito(Cost):
    @staticmethod
    def loss(a, y):
        return np.sum([y / a - log(y) / log(a) - 1 for (a, y) in zip(a, y)])

    @staticmethod
    def derivative(a, y):
        return (y + np.power(a, 2)) / np.power(a, 2)

class Contrastive(Cost):
    @staticmethod
    def loss(a, b, y, m=0, distance=euclidianDistance):
        """
            For siamese architecture:
            Evaluates a distance between two outputs with a distance function between two vectors
            :param a: output of first NN
            :param b: output of second NN
            :param y: 1 if a and b are from the same population, 0 elsewhere
            :param m: margin from where the distance between two different vectors is supposed to be enough - no need to grow more and overfit
            :param distance: distance between the two vectors
        """
        return y * distance.__func__(a, b) / 2 + (1 - y) * max(0, m - distance.__func__(a, b) ** 2 / 2)

    @staticmethod
    def derivative(a, b, y=0, m=0, distance=euclidianDistance):
        return distance.__func__(a, b)
