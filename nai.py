
import random
import math

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

        self.expectedOutput = nTimes(lambda:5,layerSizes[-1])

        self.nLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.layers = [zero(layerSizes[i]) for i in range(self.nLayers)]

        self.weights = []
        self.dws = []
        self.biases = []
        for i in range(self.nLayers - 1):
            self.weights.append(one(len(self.layers[i]) * len(self.layers[i + 1])))
        #    self.weights.append(nRandom(len(self.layers[i]) * len(self.layers[i + 1])))
            self.dws.append(zero(len(self.layers[i]) * len(self.layers[i + 1])))
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
        last_errs = []

        # Loop through each neuron in the output layer and calculate errors
        for k, n in enumerate(self.layers[-1]):
            loss = self.expectedOutput[k] - n
            err = loss * self.activation.df(n)
            last_errs.append(err)
            #dw = self.learning_rate * err * n + old_dw * self.momentum
            #print(dw)

        # Loop through each layer except output layer backwards
        for i in range(self.nLayers - 2, -1, -1):
            print(f"Layer {i}")
            layer = self.layers[i]
            weights = self.weights[i]
            dws = self.dws[i]
            rightLayerSize = len(self.layers[i + 1])

            new_last_errs = []

            print("Using last_errs", last_errs)

            # Loop through the neurons in this layer
            for k, n in enumerate(layer):
                werrSum = 0
                # Loop through the neurons in the layer to the right
                for j in range(rightLayerSize):
                    kw = k * rightLayerSize + j
                    # Get the weight between k and j (w k,j)
                    w = weights[kw]
                    e = last_errs[j]
                    werrSum += w * e

                    # Finally, calculate dw
                    dw = self.learning_rate * e * self.layers[i + 1][j] + dws[kw] * self.momentum

                    # Update the weight!
                    weights[kw] += dw

                    # Update last delta weight
                    dws[kw] = dw

                # Calculated error for this neuron's weights
                err = werrSum * self.activation.df(n)
                new_last_errs.append(err)

            last_errs = new_last_errs


    def __str__(self):
        string = ""
        for i in range(max(self.layerSizes)):
            string += "\t".join(["" if i >= len(x) else str(x[i]) for x in self.layers])
            string += "\n"
        return string


