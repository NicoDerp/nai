
import math


class ActivationFunction:
    name = "Base ActivationFunction"

    def f(x):
        return x

    def df(x):
        return 1


class Sigmoid(ActivationFunction):
    name = "Sigmoid"

    def f(x):
        return 1 / (1 + math.e**(-x))

    def df(x):
        return (math.e**x) / ((math.e**2 + 1) ** 2)


class ReLU(ActivationFunction):
    name = "ReLU"

    def f(x):
        return max(0, x)

    def df(x):
        return 0 if x <= 0 else 1


class TanH(ActivationFunction):
    name = "TanH"

    def f(x):
        return math.tanh(x)

    def df(x):
        return 1 - math.tanh(x) ** 2


