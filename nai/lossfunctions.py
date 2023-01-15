
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
        return -np.sum(yL * np.log(aL))

    @staticmethod
    @nnjit
    def df(aL, yL):
        return -yL / aL


class BinaryCrossEntropy:
    name = "Binary Cross-Entropy"

    @staticmethod
    @nnjit
    def f(aL, yL):
        return -yL * np.log(aL) - (1 - yL) * np.log(1 - aL)

    @staticmethod
    @nnjit
    def df(aL, yL):
        return (-yL / aL) + (1 - yL) / (1 - aL)

