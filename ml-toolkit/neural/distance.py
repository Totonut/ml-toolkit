# -*- coding: utf-8 -*-

import numpy as np

class Loss():
    def __call__(self, output, expected):
        raise NotImplementedError

class distest(Loss):
    def __call__(self, output, expected):
        return expected - output