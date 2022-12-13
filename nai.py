
from random import random

def zero(n):
    return [0] * n

def one(n):
    return [1] * n

def nRandom(n):
    l = []
    for i in range(n):
        l.append(random())
    return l

class ActivationFunction:
    name = "Base ActivationFunction"

    def f(x):
        return x
    
    def df(x):
        return 1

class Sigmoid(ActivationFunction):
    name = "Sigmoid"

    def f(x):
        return 1 / (1 + e**(-x))
    
    def df(x):
        return (e**x) / ((e**2 + 1) ** 2)

class ReLU(ActivationFunction):
    name = "ReLU"

    def f(x):
        return max(0, x)
    
    def df(x):
        return 0 if x <= 0 else 1

class FFNeuralNetwork:
    def __init__(self, nInput, nHidden, nOutput, activation=Sigmoid):
        self.input = zero(nInput)
        self.hidden = zero(nHidden)
        self.output = zero(nOutput)

        self.ih_weights = nRandom(nInput * nHidden)
        self.ho_weights = nRandom(nHidden * nOutput)
        
        self.ih_biases = nRandom(nInput * nHidden)
        self.ho_biases = nRandom(nHidden * nOutput)
        
        self.activation = activation
        
        print("Using activation function:", activation.name)

    def forward(self):
        pass


