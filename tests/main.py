import unittest
import timeout_decorator
import sys, inspect
import glob, os
import imp

LOCAL_TIMEOUT = 10

def getPythonFiles():
    """
        Gets all python files in tests/ directory and subdirectories
    """
    res = []
    for root, dirs, files in os.walk("./"):
        for file in files:
            if file.endswith(".py"):
                res.append(os.path.join(root, file))
    return list(map(lambda x: x[2:], res))

def getTestClasses(files):
    """
        Gets all classes inheriting from unittest.TestCase in given files
    """
    res = []
    for i in range(len(files)):
        py_mod = imp.load_source(files[i][:-3], files[i])
        for name, obj in inspect.getmembers(py_mod):
            if inspect.isclass(obj) and issubclass(obj, unittest.TestCase):
                res.append(obj)
    return res

@timeout_decorator.timeout(LOCAL_TIMEOUT)
def runTestSuite():
    """
        Launch test suite with all TestCase classes in current directory and subdirectories
    """
    test_suite = unittest.TestSuite()
    for c in getTestClasses(getPythonFiles()):
        test_suite.addTest(unittest.makeSuite(c))
    unittest.TextTestRunner().run(test_suite)

if __name__ == "__main__":
    runTestSuite()
