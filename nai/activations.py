
import math
import numpy as np

from nai.helper import *


class ActivationFunction:
    name = "Base ActivationFunction"

    def f(x):
        return x

    def df(x):
        return 1


class Sigmoid(ActivationFunction):
    name = "Sigmoid"

    @nnjit
    def f(x):
        #print(f"{x:.10f}")
        #if -x >= 710:
        #    return 0
        return 1 / (1 + np.exp(-x))

    @nnjit
    def df(x):
        a = 1 / (1 + np.exp(x))
        return a * (1 - a)


class ReLU(ActivationFunction):
    name = "ReLU"

    @nnjit
    def f(x):
        return np.maximum(0.0, x)

    @nnjit
    def df(x):
        return np.greater(x, 0.0)


class TanH(ActivationFunction):
    name = "TanH"

    @nnjit
    def f(x):
        return np.tanh(x)

    @nnjit
    def df(x):
        return 1 - np.tanh(x) ** 2

class Softmax(ActivationFunction):
    name = "Softmax"

    @nnjit
    def f(x):
        e_x = math.e ** (x - np.max(x))
        return e_x / e_x.sum(x)


