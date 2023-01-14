
import random
import numpy as np

from nai.activations import *
from nai.helper import *

import matplotlib.pyplot as plt

from numba.core import types

import pickle
from tqdm import tqdm

class MLPNeuralNetwork:
    def __init__(self, layerSizes, lossfunction, activations, learning_rate, dropout=0, adam=False):

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
        self.lossfunction = lossfunction

        self.dropout = dropout
        self.doDropout = False

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
            self.zLayers[i] = zL.copy()
            zL = self.activations[i].f(zL)
            self.layers[i + 1] = zL

    def backPropagateError(self):

        dk = self.lossfunction.df(self.layers[-1], self.expectedOutput)
        # TODO zLayers or layers??? Try both
        errs = np.multiply(dk, self.activations[-1].df(self.zLayers[-1]))
        #errs = np.multiply(dk, self.activations[-1].df(self.layers[-1]))
        self.errors[-1] = errs

        # 1
        # Only hidden layers. Input is given and output is calculated above
        # Calculate error for this layer and use weights to the right.
        for i in range(self.nLayers - 2, 0, -1):
            #wL1 = self.weights[i].reshape((self.layerSizes[i], self.layerSizes[i + 1]))
            wL1 = self.weights[i].transpose()
            eL1 = self.errors[i]
            # TODO zlayers or layyers
            eL = np.multiply(wL1.dot(eL1), self.activations[i].df(self.zLayers[i - 1]))
            #eL = np.multiply(wL1.dot(eL1), self.activations[i].df(self.layers[i]))
            self.errors[i - 1] = eL

    def gradientDescent(self):
        # 1 - 0
        for i in range(self.nLayers - 2, -1, -1):
            eL = self.errors[i] # Error for this layer
            #aL1 = self.layers[i+1]
            #aL1 = np.copy(self.layers[i]) # L-1
            aL1 = self.layers[i].reshape((-1, 1)) # 1D tranpose (1, 5) -> (5, 1)

            dw = self.learning_rate * eL * aL1
            db = self.learning_rate * eL

            #dw = dw.ravel() # Reshape from 2D to 1D

            self.weights[i] -= dw.transpose()
            self.biases[i] -= db

    def calculateLoss(self):
        return self.lossfunction.f(self.layers[-1], self.expectedOutput)

    def predict(self, data):
        assert data.shape == self.layers[0].shape
        self.layers[0] = data
        self.forwardPropagate()
        return self.layers[-1]

    def train(self, dataset, epochs=10, batch_size=32):
        if batch_size > dataset.size:
            raise ValueError(f"Batch size is greater than dataset size")

        #if dataset.shape != (1, self.net.layerSizes[0]):
        #    raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        self.doDropout = True

        #dataset.useSet(SetTypes.Train)

        nBatches = math.ceil(dataset.size / batch_size)
        print(f"Doing {nBatches} batches per epoch")

        lossArray = np.empty(epochs * nBatches)
        #lossArray = np.empty(epochs)
        #accuracyArray = np.empty(epochs*nBatches)

        batchCount = 0

        for epoch in tqdm(range(epochs)):
            print(f"Epoch {epoch+1}/{epochs}")

            dataset.shuffle()

            for batch in tqdm(range(nBatches), leave=False):
                averageError = np.array([np.zeros(self.layerSizes[i + 1]) for i in range(self.nLayers - 1)], dtype=object)
    
                averageLoss = 0
                #averageAcc = 0
            
                samples = dataset.retrieveBatch(batch_size)
                #print([(sample.data, sample.output) for sample in samples])
            
                for sample in samples:
                    #print("Sample")
                    #print("\nUsing data:", sample.data)
                    self.layers[0] = sample.data
                    self.forwardPropagate()
                    #print("Got answ:", self.net.layers[-1])
            
                    self.expectedOutput = sample.output
                    self.backPropagateError()
                    averageError = np.add(averageError, self.errors, dtype=object)
            
                    loss = self.calculateLoss()
                    averageLoss += loss
            
                    #pred = np.argmax(self.layers[-1])
                    #prob = self.layers[-1][pred]
            
                    #averageAcc += 1 if biggest_i == list(sample.output).index(1) else 0
            
                # Calculate average

                averageError /= len(samples)

                self.errors = averageError
                self.gradientDescent()

                #accuracyArray[batchCount] = acc
                lossArray[batchCount] = averageLoss / len(samples)

                batchCount += 1

            # Debug
            #lossArray.append(averageLoss / batch_size)

            # Optimizations
            #if self.adam:
            #    Mht = Mt / (1 / self.bias1 ** epoch)
            #    self.net.learning_rate /= math.sqrt(epoch)

        self.doDropout = False

        print(lossArray)

        # Debug
        #plt.plot(range(epochs * nBatches), lossArray, label="Loss")
        plt.plot(range(epochs*nBatches), lossArray, label="Loss")
        #plt.plot(range(epochs * nBatches), accuracyArray, label="Accuracy")
        plt.grid()
        plt.legend()
        plt.show()

    def test(self, dataset, nSamples=500):
        #if dataset.shape != (1, self.net.layerSizes[0]):
        #    raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        #dataset.useSet(SetTypes.Train)

        dataset.shuffle()

        averageLoss = 0
        nSamples = min(nSamples, dataset.size)

        # Just in case
        self.doDropout = False

        samples = dataset.retrieveBatch(nSamples)

        for sample in samples:
            self.layers[0] = sample.data
            self.forwardPropagate()

            self.expectedOutput = sample.output
            averageLoss += self.calculateLoss()

        averageLoss /= nSamples
        print(f"Average loss for {nSamples} samples is {averageLoss}")

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.layerSizes, f)
            pickle.dump(self.lossfunction, f)
            pickle.dump(self.activations, f)
            pickle.dump(self.learning_rate, f)
            pickle.dump(self.dropout, f)

            pickle.dump(self.weights, f)
            pickle.dump(self.biases, f)

    @classmethod
    def load(cls, filename):
        self = cls.__new__(cls)

        with open(filename, "rb") as f:
            self.layerSizes = pickle.load(f)
            self.lossfunction = pickle.load(f)
            self.activations = pickle.load(f)
            self.learning_rate = pickle.load(f)
            self.dropout = pickle.load(f)

            self.weights = pickle.load(f)
            self.biases = pickle.load(f)

        self.doDropout = False
        self.adam = False

        # Adam optimizer parameters
        if self.adam:
            self.momentum = 0.9
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 10**-8

        self.expectedOutput = np.zeros(self.layerSizes[-1])

        self.nLayers = len(self.layerSizes)
        self.layerSizes = self.layerSizes
        self.layers = [np.zeros(self.layerSizes[i], dtype=np.float64) for i in range(self.nLayers)]
        self.zLayers = [np.zeros(self.layerSizes[i + 1], dtype=np.float64) for i in range(self.nLayers - 1)] # The same as layers but every neuron is before activation

        self.errors = [np.zeros(self.layerSizes[i + 1]) for i in range(self.nLayers - 1)]

        return self

    def __str__(self):
        string = "Input          "
        string += "".join(f"Hidden {i}{' '*(8-len(str(i)))}" for i in range(self.nLayers-2))
        string += "Output\n"

        for i in range(max(self.layerSizes)):
            string += "        ".join(["       " if i >= len(x) else f"{x[i]:.5f}" for x in self.layers])
            string += "\n"
        return string


