
import math
from numba import njit
import numpy as np

class ActivationFunction:
    name = "Base ActivationFunction"

    def f(x):
        return x

    def df(x):
        return 1


class Sigmoid(ActivationFunction):
    name = "Sigmoid"

    @njit(fastmath=True)
    def f(x):
        #print(f"{x:.10f}")
        #if -x >= 710:
        #    return 0
        return 1 / (1 + math.e**(-x))

    @njit(fastmath=True)
    def df(x):
        return (math.e**x) / ((math.e**2 + 1) ** 2)


class ReLU(ActivationFunction):
    name = "ReLU"

    @njit(fastmath=True)
    def f(x):
        return max(0, x)

    def df(x):
        return 0 if x <= 0 else 1


class TanH(ActivationFunction):
    name = "TanH"

    @njit(fastmath=True)
    def f(x):
        return math.tanh(x)

    @njit(fastmath=True)
    def df(x):
        return 1 - math.tanh(x) ** 2

class Softmax(ActivationFunction):
    name = "Softmax"

    @njit(fastmath=True)
    def f(x):
        e_x = math.e ** (x - np.max(x))
        return e_x / e_x.sum(x)


