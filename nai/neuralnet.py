
import random
import math

from nai.activations import *

def nTimes(func, n):
    l = []
    for i in range(n):
        l.append(func())
    return l

def zero(n):
    return [0] * n

def one(n):
    return [1] * n

def nRandom(n):
    l = []
    for i in range(n):
        #l.append(random.random() * 2 - 1)
        l.append(random.random())
    return l

class MLPNeuralNetwork:
    def __init__(self, layerSizes, activation=Sigmoid):

        self.momentum = 1.0
        self.learning_rate = 0.5

        if len(layerSizes) < 3:
            raise ValueError("A multilayer perceptron must consist of an input layer, atleast one hidden layer and an ouuput layer.")

        self.expectedOutput = nTimes(lambda:5,layerSizes[-1])

        self.nLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.layers = [zero(layerSizes[i]) for i in range(self.nLayers)]
        self.zLayers = [zero(layerSizes[i]) for i in range(1, self.nLayers)] # The same as layers but every neuron is before activation

        self.weights = []
        self.biases = []
        self.lastErrors = []
        for i in range(self.nLayers - 1):
            #self.weights.append(one(len(self.layers[i]) * len(self.layers[i + 1])))
            self.weights.append(nRandom(len(self.layers[i]) * len(self.layers[i + 1])))
            #self.biases.append(zero(len(self.layers[i+1])))
            self.biases.append(nRandom(len(self.layers[i + 1])))
            self.lastErrors.append(zero(len(self.layers[i + 1])))

        self.activation = activation

        print("Using activation function:", activation.name)

    def forwardPropagation(self):
        # Loop through input and hidden layers
        for i in range(self.nLayers - 1):
            layer1 = self.layers[i]
            layer2 = self.layers[i + 1]
            weights = self.weights[i]
            biases = self.biases[i]

            # Loop through the neurons on the second layer
            for j in range(len(layer2)):
                s = 0

                # Loop through neurons on the first layer
                for k, n in enumerate(layer1):
                    kw = k * len(layer2) + j
                    w = weights[kw]
                    s += w * n

                s += biases[j]
                self.zLayers[i][j] = s
                s = self.activation.f(s)
                layer2[j] = s

    def backPropagation(self):
        # Loop through each neuron in the output layer and calculate errors
        for k, n in enumerate(self.layers[-1]):
            dk = n - self.expectedOutput[k]
            err = dk * self.activation.df(self.zLayers[-1][k])
            self.lastErrors[-1][k] = err

        # Loop through each layer except output layer backwards
        for i in range(self.nLayers - 1, 0, -1):
            layer = self.layers[i]

            # Loop through the neurons in the layer to the left
            for j in range(len(self.layers[i - 1])):
                werrSum = 0
                # Loop through the neurons in this layer
                for k in range(len(layer)):
                    kw = j * len(self.layers[i]) + k # left*sizeof(right) + right

                    # Get the weight between input (j) and output (k) (w k,j)
                    w = self.weights[i - 1][kw]
                    e = self.lastErrors[i - 1][k] # Get the error from neuron in this layer
                    werrSum += w * e

                    # Finally, calculate dw
                    dw = -self.learning_rate * self.layers[i - 1][j] * e
                    db = -self.learning_rate * e

                    # Update the weight between this layer and the one to the left
                    self.weights[i - 1][kw] += dw

                    # Update the bias for the neuron on this layer
                    self.biases[i - 1][k] += db

                # Calculated error for neuron on the layer to the left
                if i > 1:
                    err = werrSum * self.activation.df(self.zLayers[i - 2][j])
                    self.lastErrors[i - 2][j] = err

    def calculateLoss(self):
        E = 0
        for i, n in enumerate(self.layers[-1]):
            E += (self.expectedOutput[i] - n) ** 2

        return E / len(self.layers[-1])

    def __str__(self):
        string = "Input          "
        string += "".join(f"Hidden {i}{' '*(8-len(str(i)))}" for i in range(self.nLayers-2))
        string += "Output\n"

        for i in range(max(self.layerSizes)):
            string += "        ".join(["       " if i >= len(x) else f"{x[i]:.5f}" for x in self.layers])
            string += "\n"
        return string

