
import random
import math
import numpy as np

from numba import njit

from nai.activations import *


def _forwardPropagate(layers, weights, biases, zLayers, activation):
    # Loop through input and hidden layers
    for i in range(len(layers) - 1):
        #print(f"Layer {i}")

        # Loop through the neurons on the second layer
        for j in range(len(layers[i + 1])):
            s = 0

            # Loop through neurons on the first layer
            for k in range(len(layers[i])):
                n = layers[i][k]
                kw = k * len(layers[i + 1]) + j
                w = weights[i][kw]
                s += w * n

            s += biases[i][j]
            zLayers[i][j] = s

            s = activation(s)
            layers[i + 1][j] = s


class MLPNeuralNetwork:
    def __init__(self, layerSizes, learning_rate, activation=Sigmoid, adam=False):

        self.learning_rate = learning_rate

        self.adam = adam

        # Adam optimizer parameters
        if self.adam:
            self.momentum = 0.9
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 10**-8

        if len(layerSizes) < 3:
            raise ValueError("A multilayer perceptron must consist of an input layer, atleast one hidden layer and an ouuput layer.")

        self.expectedOutput = np.zeros(layerSizes[-1])

        self.nLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.layers = np.array([np.zeros(layerSizes[i]) for i in range(self.nLayers)], dtype=object)
        self.zLayers = np.array([np.zeros(layerSizes[i]) for i in range(1, self.nLayers)], dtype=object) # The same as layers but every neuron is before activation

        self.weights = np.array([np.random.uniform(size=layerSizes[i] * layerSizes[i + 1]) for i in range(self.nLayers - 1)], dtype=object)
        #self.weights = np.array([np.zeros(layerSizes[i] * layerSizes[i + 1]) for i in range(self.nLayers - 1)])
        self.biases = np.array([np.random.uniform(size=layerSizes[i + 1]) for i in range(self.nLayers - 1)], dtype=object)
        #self.biases = np.array([np.zeros(layerSizes[i + 1]) for i in range(self.nLayers - 1)])

        self.activation = activation

        print("Using activation function:", activation.name)

    def forwardPropagate(self):
        _forwardPropagate(self.layers, self.weights, self.biases, self.zLayers, self.activation.f)

    def calcErrors(self):
        lastErrors = np.array([np.zeros(self.layerSizes[i + 1]) for i in range(self.nLayers - 1)], dtype=object)

        # Loop through each neuron in the output layer and calculate errors
        for k, n in enumerate(self.layers[-1]):
            dk = n - self.expectedOutput[k]
            err = dk * self.activation.df(self.zLayers[-1][k])
            lastErrors[-1][k] = err

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
                    e = lastErrors[i - 1][k] # Get the error from neuron in this layer
                    werrSum += w * e

                # Calculated error for neuron on the layer to the left
                if i > 1:
                    err = werrSum * self.activation.df(self.zLayers[i - 2][j])
                    lastErrors[i - 2][j] = err

        return lastErrors

    def backPropagate(self, errorList):
        # Loop through each layer except output layer backwards
        for i in range(self.nLayers - 1, 0, -1):
            layer = self.layers[i]

            # Loop through the neurons in the layer to the left
            for j in range(len(self.layers[i - 1])):
                # Loop through the neurons in this layer
                for k in range(len(layer)):
                    kw = j * len(self.layers[i]) + k # left*sizeof(right) + right

                    e = errorList[i - 1][k] # Get the error from neuron in this layer

                    # Finally, calculate dw and db
                    dw = -self.learning_rate * self.layers[i - 1][j] * e
                    db = -self.learning_rate * e

                    # Update the weight between this layer and the one to the left
                    self.weights[i - 1][kw] += dw

                    # Update the bias for the neuron on this layer
                    self.biases[i - 1][k] += db

    def calculateLoss(self):
        E = 0
        for i, n in enumerate(self.layers[-1]):
            #print(f"i {i}, n {n}")
            E += (n - self.expectedOutput[i]) ** 2
            #print(f"E {E}")

        return E / len(self.layers[-1])

    def __str__(self):
        string = "Input          "
        string += "".join(f"Hidden {i}{' '*(8-len(str(i)))}" for i in range(self.nLayers-2))
        string += "Output\n"

        for i in range(max(self.layerSizes)):
            string += "        ".join(["       " if i >= len(x) else f"{x[i]:.5f}" for x in self.layers])
            string += "\n"
        return string


