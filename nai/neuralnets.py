
import random
import math
import numpy as np

from nai.activations import *
from nai.helper import *


#@numba.njit("f8(f8[:], f8[:])", fastmath=True)
@nnjit
def _calculateLoss(outputLayer, expectedOutput):
    E = 0.0
    for i in range(len(outputLayer)):
        #print(f"i {i}, n {n}")
        E += (expectedOutput[i] - outputLayer[i]) ** 2
        #print(f"E {E}")

    return E / 2

class MLPNeuralNetwork:
    def __init__(self, layerSizes, learning_rate, activations=[], adam=False):

        if len(layerSizes) < 3:
            raise ValueError("A multilayer perceptron must consist of an input layer, atleast one hidden layer and an output layer.")

        if not isinstance(activations, list):
            if issubclass(activations, ActivationFunction):
                self.activations = [activations] * (len(layerSizes) - 1)
            else:
                raise ValueError("Activations needs to be either an ActivationFunction or a list of ActivationFunctions.")
        elif len(layerSizes) - 1 != len(activations):
            raise ValueError("Activations must be 1 less than the amount of layers.")
        else:
            self.activations = activations

        self.learning_rate = learning_rate

        self.adam = adam

        # Adam optimizer parameters
        if self.adam:
            self.momentum = 0.9
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 10**-8

        self.expectedOutput = np.zeros(layerSizes[-1])

        self.nLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.layers = [np.zeros(layerSizes[i], dtype=np.float64) for i in range(self.nLayers)]
        self.zLayers = [np.zeros(layerSizes[i + 1], dtype=np.float64) for i in range(self.nLayers - 1)] # The same as layers but every neuron is before activation

        self.errors = [np.zeros(self.layerSizes[i + 1]) for i in range(self.nLayers - 1)]

        # Random initialized
        self.weights = [np.random.uniform(size=(layerSizes[i + 1], layerSizes[i])) for i in range(self.nLayers - 1)]
        self.biases = [np.random.uniform(size=layerSizes[i + 1]) for i in range(self.nLayers - 1)]

        # Zero initialized
        #self.weights = [np.zeros(shape=(layerSizes[i + 1], layerSizes[i])) for i in range(self.nLayers - 1)]
        #self.biases = [np.zeros(shape=layerSizes[i + 1]) for i in range(self.nLayers - 1)]

    def forwardPropagate(self):
        # 0 - (len(self.nLayers) - 1)
        for i in range(self.nLayers - 1):
            #wL = self.weights[i].reshape((self.layerSizes[i + 1], self.layerSizes[i]))
            wL = self.weights[i]
            aL = self.layers[i]
            zL = wL.dot(aL)
            zL += self.biases[i]
            #self.zLayers[i] = zL.copy()
            zL = self.activations[i].f(zL)
            self.layers[i + 1] = zL

        # Loop through input and hidden layers
        #for i in range(len(layers) - 1):
            #print(f"Layer {i}")

            # Loop through the neurons on the second layer
            #for j in range(len(layers[i + 1])):
                #s = 0

                # Loop through neurons on the first layer
                #for k in range(len(layers[i])):
                    #n = layers[i][k]
                    #kw = k * len(layers[i + 1]) + j
                    #w = weights[i][kw]
                    #s += w * n

                #s += biases[i][j]
                #zLayers[i][j] = s

                #s = activation(s)
                #layers[i + 1][j] = s

    def backPropagateError(self):

        dk = self.layers[-1] - self.expectedOutput
        # TODO zLayers or layers??? Try both
        #errs = np.multiply(dk, self.activations[-1].df(self.zLayers[-1]))
        errs = np.multiply(dk, self.activations[-1].df(self.layers[-1]))
        self.errors[-1] = errs

        # Loop through each neuron in the output layer and calculate errors
        #for k, n in enumerate(self.layers[-1]):
        #    dk = n - self.expectedOutput[k]
        #    err = dk * self.activation.df(self.zLayers[-1][k])
        #    lastErrors[-1][k] = err

        # 1
        # Only hidden layers. Input is given and output is calculated above
        # Calculate error for this layer and use weights to the right.
        for i in range(self.nLayers - 2, 0, -1):
            #wL1 = self.weights[i].reshape((self.layerSizes[i], self.layerSizes[i + 1]))
            wL1 = self.weights[i].transpose()
            eL1 = self.errors[i]
            #eL = np.multiply(wL1.dot(eL1), self.activations[i].df(self.zLayers[i - 1]))
            eL = np.multiply(wL1.dot(eL1), self.activations[i].df(self.layers[i]))
            self.errors[i - 1] = eL

        # Loop through each layer except output layer backwards
        #for i in range(self.nLayers - 1, 0, -1):
        #    layer = self.layers[i]

            # Loop through the neurons in the layer to the left
            #for j in range(len(self.layers[i - 1])):
                #werrSum = 0
                # Loop through the neurons in this layer
                #for k in range(len(layer)):
                    #kw = j * len(self.layers[i]) + k # left*sizeof(right) + right

                    # Get the weight between input (j) and output (k) (w k,j)
                    #w = self.weights[i - 1][kw]
                    #e = lastErrors[i - 1][k] # Get the error from neuron in this layer
                    #werrSum += w * e

                # Calculated error for neuron on the layer to the left
                #if i > 1:
                    #err = werrSum * self.activation.df(self.zLayers[i - 2][j])
                    #lastErrors[i - 2][j] = err

    def gradientDescent(self):
        # 1 - 0
        for i in range(self.nLayers - 2, -1, -1):
            eL = self.errors[i] # Error for this layer
            #aL1 = self.layers[i+1]
            aL1 = np.copy(self.layers[i]) # L-1
            aL1 = aL1.reshape((-1, 1)) # 1D tranpose (1, 5) -> (5, 1)

            #dw = self.learning_rate * eL * aL1
            dw = self.learning_rate * eL * aL1
            db = self.learning_rate * eL

            #dw = dw.ravel() # Reshape from 2D to 1D

            self.weights[i] -= dw.transpose()
            self.biases[i] -= db

        # Loop through each layer except output layer backwards
        #for i in range(self.nLayers - 1, 0, -1):
            #layer = self.layers[i]

            # Loop through the neurons in the layer to the left
            #for j in range(len(self.layers[i - 1])):
                # Loop through the neurons in this layer
                #for k in range(len(layer)):
                    #kw = j * len(self.layers[i]) + k # left*sizeof(right) + right

                    #e = errorList[i - 1][k] # Get the error from neuron in this layer

                    # Finally, calculate dw and db
                    #dw = -self.learning_rate * self.layers[i - 1][j] * e
                    #db = -self.learning_rate * e

                    # Update the weight between this layer and the one to the left
                    #self.weights[i - 1][kw] += dw

                    # Update the bias for the neuron on this layer
                    #self.biases[i - 1][k] += db

    def calculateLoss(self):
        return _calculateLoss(self.layers[-1], self.expectedOutput)

    def __str__(self):
        string = "Input          "
        string += "".join(f"Hidden {i}{' '*(8-len(str(i)))}" for i in range(self.nLayers-2))
        string += "Output\n"

        for i in range(max(self.layerSizes)):
            string += "        ".join(["       " if i >= len(x) else f"{x[i]:.5f}" for x in self.layers])
            string += "\n"
        return string


