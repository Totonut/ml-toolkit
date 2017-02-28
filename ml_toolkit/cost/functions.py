from math import *
from abc import *
import numpy as np

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

    @staticmethod
    def checkVectors(a, y):
        assert len(a) == len(y), "Wrong arguments in loss function: lengths mismatch"
        if not isinstance(a, (np.ndarray, np.generic)):
            a = np.array(a)
        if not isinstance(y, (np.ndarray, np.generic)):
            y = np.array(y)
        return a, y

class Euclidian(Cost):
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return sqrt(sum([(a - y) ** 2 for (a, y) in zip(a, y)]))

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        return 2 * (a - y)

class Cosine(Cost):
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return 1 - sum([a * y for (a, y) in zip(a, y)]) / (sqrt(sum([a ** 2 for a in a])) * sqrt(sum([y ** 2 for y in y])))

    @staticmethod
    def derivative(a, y):
        return Cosine.loss(a, y) * (a / np.power(a, 2)) - y / (sqrt(sum([a ** 2 for a in a])) * sqrt(sum([y ** 2 for y in y])))

class CrossEntropy(Cost):
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return -np.sum([y * log(a) + (1 - y) * log(1 - a) for (a, y) in zip(a, y)]) / len(a)

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        return (a - y) / (a * (1 - a))


class Quadratic(Cost):
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        assert len(a) == len(y), "wrong arguments in quadratic loss function: lengths mismatch"
        return np.sum([(a - y) ** 2 for (a, y) in zip(a, y)]) / (2 * len(a))

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        assert len(a) == len(y), "wrong arguments in quadratic loss function: lengths mismatch"
        return a - y

class Exponential(Cost):
    @staticmethod
    def loss(a, y, l=1):
        a, y = Cost.checkVectors(a, y)
        return l * exp((1 / l) * np.sum([(a - y) ** 2 for (a, y) in zip(a, y)]))

    @staticmethod
    def derivative(a, y, l=1):
        a, y = Cost.checkVectors(a, y)
        return (2 / l) * (a - y) * Exponential.loss(a, y, l)

class Hellinger(Cost):
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return (1 / sqrt(2)) * np.sum([(sqrt(a) - sqrt(y)) ** 2 for (a, y) in zip(a, y)])

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        return (np.sqrt(a) - np.sqrt(y)) / (sqrt(2) * np.sqrt(a))

class KullbackLeibler(Cost):
    """
        This evaluates the loss of information when y is used to approximate a
    """
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return np.sum([a * log(a / y) for (a, y) in zip(a, y)])

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        return y / a

class GeneralizedKullbackLeibler(Cost):
    """
        This evaluates the loss of information when y is used to approximate a
    """
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return np.sum([a * log(a / y) for (a, y) in zip(a, y)]) - np.sum(y) + np.sum(a)

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        return (y + a) / a
        

class ItakuraSaito(Cost):
    @staticmethod
    def loss(a, y):
        a, y = Cost.checkVectors(a, y)
        return np.sum([y / a - log(y / a) - 1 for (a, y) in zip(a, y)])

    @staticmethod
    def derivative(a, y):
        a, y = Cost.checkVectors(a, y)
        return (y + np.power(a, 2)) / np.power(a, 2)

class Contrastive(Cost):
    @staticmethod
    def loss(a, b, y, m=0, distance=Euclidian):
        a, b = Cost.checkVectors(a, b)
        """
            For siamese architecture:
            Evaluates a distance between two outputs with a distance function between two vectors
            :param a: output of first NN
            :param b: output of second NN
            :param y: 0 if a and b are from the same population, 1 elsewhere
            :param m: margin from where the distance between two different vectors is supposed to be enough - no need to grow more and overfit
            :param distance: distance between the two vectors
        """
        return (1 - y) * distance.loss(a, b) / 2 + y * max(0, m - distance.loss(a, b) ** 2 / 2)

    @staticmethod
    def derivative(a, b, y, m=0, distance=Euclidian):
        a, b = Cost.checkVectors(a, b)
        if y == 0:
            return distance.loss(a, b) * distance.derivative(a, b)
        return (distance.loss(a, b) - m) * distance.derivative(a, b)
