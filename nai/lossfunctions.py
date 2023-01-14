
import numpy as np
import math

from nai.helper import *

class LossFunction:
    name = "Base LossFunction"

    @staticmethod
    def f(aL, yL):
        return 0

    @staticmethod
    def df(aL, yL):
        return 0

class MSE:
    name = "Mean-squared error"

    @staticmethod
    @nnjit
    def f(aL, yL):
        return np.sum((yL - aL) ** 2) * 0.5

    @staticmethod
    @nnjit
    def df(aL, yL):
        return aL - yL


class CrossEntropy:
    name = "Cross-Entropy"

    @staticmethod
    @nnjit
    def f(aL, yL):
        return -np.sum(yL * np.log10(aL))

    @staticmethod
    @nnjit
    def df(aL, yL):
        return -yL / (aL * np.log(math.e))

