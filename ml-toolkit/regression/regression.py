import numpy as np
from numpy.linalg import inv
from statistics import *

class Regression:
    @staticmethod
    def linear_regression(y, x):
        y = np.array(y)
        x = np.array(x)
        return np.dot(np.dot(inv(np.dot(x.transpose(), x)), x.transpose()), y)

    @staticmethod
    def polynomial_regression(y, x):
        pass

    @staticmethod
    def logistic_regression(y, x):
        pass

if __name__ == "__main__":
    a = Regression.linear_regression([0, 1, 0], [[1, 0], [1, 1], [1, 2]])
    print(a)
