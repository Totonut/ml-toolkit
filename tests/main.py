import unittest
from genetic_unit import TestGenetic

test_suite = unittest.TestSuite()
test_suite.addTest(unittest.makeSuite(TestGenetic))
unittest.TextTestRunner().run(test_suite)
