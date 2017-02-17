import unittest
import random as rnd
import math
from ml_toolkit import *

class TestCost(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCost, self).__init__(*args, **kwargs)
        self.sample_size = 5
        self.close_to_one = 0.99
        self.close_to_zero = 0.01
        self.close_to_zero_step = 0.001
        self.a = []
        self.b = []
        self.c = []
        self.x = []
        for _ in range(self.sample_size):
            binary = rnd.randint(0, 1)
            self.b.append(binary)
            self.c.append(1 - binary)
            if binary:
                self.a.append(rnd.randrange(self.close_to_one * 1000, 1000, 1) / 1000)
            else:
                self.a.append(rnd.randrange(1, self.close_to_zero * 1000 + 1, 1) / 1000)
            zero_to_one = rnd.random()
            self.x.append(zero_to_one)
            zero_to_one = rnd.random()
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)
        self.x = np.array(self.x)

    def testCrossEntropy(self):
        self.assertAlmostEqual(CrossEntropy.loss(self.a, self.b), 0, places=1)
        self.assertTrue(len(CrossEntropy.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertAlmostEqual(math.fabs(CrossEntropy.derivative(self.a, self.b)[i]), 1, places=1)

    def testQuadratic(self):
        self.assertEqual(Quadratic.loss(self.x, self.x), 0)
        self.assertTrue(len(Quadratic.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(Quadratic.derivative(self.x, self.x)[i], 0)

    def testExponential(self):
        self.assertEqual(Exponential.loss(self.x, self.x), 1)
        self.assertTrue(len(Exponential.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(Exponential.derivative(self.x, self.x)[i], 0)

    def testHellinger(self):
        self.assertEqual(Hellinger.loss(self.x, self.x), 0)
        self.assertTrue(len(Hellinger.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(Hellinger.derivative(self.x, self.x)[i], 0)

    def testKullbackLeibler(self):
        self.assertEqual(KullbackLeibler.loss(self.x, self.x), 0)
        self.assertTrue(len(KullbackLeibler.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(KullbackLeibler.derivative(self.x, self.x)[i], 1)

    def testGeneralizedKullbackLeibler(self):
        self.assertEqual(GeneralizedKullbackLeibler.loss(self.x, self.x), 0)
        self.assertTrue(len(GeneralizedKullbackLeibler.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(GeneralizedKullbackLeibler.derivative(self.x, self.x)[i], 2)

    def testItakuraSaito(self):
        self.assertEqual(ItakuraSaito.loss(self.x, self.x), 0)
        self.assertTrue(len(ItakuraSaito.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertAlmostEqual(ItakuraSaito.derivative(self.x, self.x)[i], (1 / self.x[i]) + 1)

    def testContrastive(self):
        self.assertEqual(Contrastive.loss(self.x, self.x, 0), 0)
        self.assertTrue(len(Contrastive.derivative(self.x, self.x, 0)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(Contrastive.derivative(self.x, self.x, 0)[i], 0)

    def testEuclidian(self):
        self.assertEqual(Euclidian.loss(self.x, self.x), 0)
        self.assertTrue(len(Euclidian.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertEqual(Euclidian.derivative(self.x, self.x)[i], 0)

    def testCosine(self):
        self.assertAlmostEqual(Cosine.loss(self.x, self.x), 0, places=1)
        self.assertTrue(len(Cosine.derivative(self.x, self.x)) > 0)
        for i in range(self.sample_size):
            self.assertAlmostEqual(Euclidian.derivative(self.x, self.x)[i], 0)
