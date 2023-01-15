
import math
import numpy as np

from nai.helper import *


class ActivationFunction:
    name = "Base ActivationFunction"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def df(x):
        return 1


class Sigmoid(ActivationFunction):
    name = "Sigmoid"

    @staticmethod
    @nnjit
    def f(x):
        #print(f"{x:.10f}")
        #if -x >= 710:
        #    return 0
        return 1 / (1 + np.exp(-x))

    @staticmethod
    @nnjit
    def df(x):
        a = 1 / (1 + np.exp(x))
        return a * (1 - a)


class ReLU(ActivationFunction):
    name = "ReLU"

    @staticmethod
    @nnjit
    def f(x):
        return np.maximum(0.0, x)

    @staticmethod
    @nnjit
    def df(x):
        return np.greater(x, 0.0)


class TanH(ActivationFunction):
    name = "TanH"

    @staticmethod
    @nnjit
    def f(x):
        return np.tanh(x)

    @staticmethod
    @nnjit
    def df(x):
        return 1 - np.tanh(x) ** 2


class Softmax(ActivationFunction):
    name = "Softmax"

    @staticmethod
    @nnjit
    def f(x):
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

    @staticmethod
    @nnjit
    def df(x): # TODO
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

