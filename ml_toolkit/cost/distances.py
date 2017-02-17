from math import *

"""
    Distances functions to evaluate a metric between two vectors
    For each function, a and b are the two vectors
"""

@staticmethod
def euclidianDistance(a, b):
    assert len(a) == len(b), "wrong arguments in euclidian distance function: lengths mismatch"
    return sqrt(sum([(a - b) ** 2 for (a, b) in zip(a, b)]))

@staticmethod
def cosineDistance(a, b):
    assert len(a) == len(b), "wrong arguments in cosine distance function: lengths mismatch"
    return 1 - sum([a * b for (a, b) in zip(a, b)]) / (sqrt(sum([a ** 2 for a in a])) * sqrt(sum([b ** 2 for b in b])))
