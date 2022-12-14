
import random
import math

def nTimes(n, func):
    l = []
    for i in range(n):
        l.append(func())
    return l

def zero(n):
    return [0] * n

def one(n):
    return [1] * n

def nRandom(n):
    return nTimes(n, random.random)

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

class MLPNeuralNetwork:
    def __init__(self, layerSizes, activation=Sigmoid):

        self.momentum = 1.0
        self.learning_rate = 1.0

        if len(layerSizes) < 3:
            raise ValueError("A multilayer perceptron must consist of an input layer, atleast one hidden layer and an ouuput layer.")

        self.expectedOutput = one(layerSizes[-1])

        self.nLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.layers = [zero(layerSizes[i]) for i in range(self.nLayers)]

        self.weights = []
        for i in range(self.nLayers - 1):
            self.weights.append(one(len(self.layers[i]) * len(self.layers[i + 1])))
        #    self.weights.append(nRandom(len(self.layers[i]) * len(self.layers[i + 1])))

        self.biases = []
        for i in range(self.nLayers - 1):
            self.biases.append(zero(len(self.layers[i+1])))

        self.activation = activation

        print("Using activation function:", activation.name)

    def forwardPropagation(self):
        # Loop through each layer
        for i in range(self.nLayers - 1):
            layer1 = self.layers[i]
            layer2 = self.layers[i + 1]
            weights = self.weights[i]
            biases = self.biases[i]

            # Loop through the neurons on the second layer (the one we're calculating)
            for j in range(len(layer2)):
                s = 0

                # Loop through weights and neurons from the first layer
                for n1, w in zip(layer1, weights[j*len(layer1):]):
                    s += w * n1

                s += biases[j]
                s = self.activation.f(s)
                layer2[j] = s

    def backPropagation(self):
        # Loop through each layer backwards
        for i in range(self.nLayers - 1, 0, -1):
            layer = self.layers[i]
            leftLayerSize = len(self.layers[-1])
            old_dw = 0 # TODO
            # Loop through each neuron in the current layer
            for j, n in enumerate(layer):
                # Check if it is the output layer
                if i == self.nLayers - 1:
                    loss = self.expectedOutput[j] - n
                    err = loss * self.activation.df(n)
                    dw = self.learning_rate * err * n + old_dw * self.momentum
                    old_dw = dw
                else:
                    loss = self.expectedOutput[j+1] - n
                    err = loss * self.expectedOutput[j+1]
                    dw = self.learning_rate * err * n + old_dw * self.momentum

    def __str__(self):
        string = ""
        for i in range(max(self.layerSizes)):
            string += "\t".join(["" if i >= len(x) else str(x[i]) for x in self.layers])
            string += "\n"
        return string


