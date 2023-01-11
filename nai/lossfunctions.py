
import numpy as np
import math

from nai.helper import *

class LossFunction:
    name = "Base LossFunction"

    def f(aL, yL):
        return 0

    def df(aL, yL):
        return 0

class MSE:
    name = "Mean-squared error"

    @nnjit
    def f(aL, yL):
        return np.sum((yL - aL) ** 2) * 0.5

    @nnjit
    def df(aL, yL):
        return aL - yL


class CrossEntropy:
    name = "Cross-Entropy"

    @nnjit
    def f(aL, yL):
        return -np.sum(yL * np.log10(aL))

    @nnjit
    def df(aL, yL):
        return -yL / (aL * np.log(math.e))

