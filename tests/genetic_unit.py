import unittest
import random
from genetic import *

def remove(array, e):
    """
        Redefines removing function so that it returns the array
    """
    array.remove(e)
    return array

class TestGenetic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGenetic, self).__init__(*args, **kwargs)
        self.fitness = lambda x : x
        self.data = shuffle(list(range(1000)))
        self.sample_size = 5

    def testShuffle(self):
        data = RandomSelection.shuffle(self.data, self.sample_size)
        self.assertTrue(len(data) == self.sample_size)
        for e in data:
            self.assertTrue(e not in remove(data[:], e))
            
    def testReservoirSampling(self):
        data = RandomSelection.reservoirSampling(self.data, self.sample_size)
        self.assertTrue(len(data) == self.sample_size)
        for e in data:
            self.assertTrue(e not in remove(data[:], e))

    def testRank(self):
        data = WeightedSelection.rank(self.data, self.fitness, self.sample_size)
        self.assertTrue(len(data) == self.sample_size)
        self.assertTrue(sum(data) >= len(data) * (len(data) + 1) + 2)

    def testWheel(self):
        data = WeightedSelection.wheel(self.data, self.fitness, self.sample_size)
        self.assertTrue(len(data) == self.sample_size)
        self.assertTrue(sum(data) >= len(data) * (len(data) + 1) + 2)

    def testEugenism(self):
        data = WeightedSelection.eugenism(self.data, self.fitness, self.sample_size)
        self.assertTrue(len(data) == self.sample_size)
        self.assertTrue(sum(data) >= len(data) * (len(data) + 1) + 2)

    def testTournament(self):
        data = WeightedSelection.tournament(self.data, self.fitness, len(self.data))
        self.assertTrue(len(data) == len(self.data))
        self.assertTrue(sum(data) >= len(data) * (len(data) + 1) / 2)
