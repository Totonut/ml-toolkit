from math import *

"""
    Distances functions to evaluate a metric between two vectors
    For each function, a and b are the two vectors
"""

def euclidian_distance(a, b):
    assert len(a) == len(b), "wrong arguments in euclidian distance function: lengths mismatch"
    return sqrt(sum([(a - b) ** 2 for (a, b) in zip(a, b)]))

def cosine_distance(a, b):
    assert len(a) == len(b), "wrong arguments in cosine distance function: lengths mismatch"
    return 1 - sum([a * b for (a, b) in zip(a, b)]) / (sqrt(sum([a ** 2 for a in a])) * sqrt(sum([b ** 2 for b in b])))

a = [2, 3, 3]
b = [2, 3, 3]
print(euclidian_distance(a, b))
print(cosine_distance(a, b))
