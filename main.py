#!/usr/bin/python3

from nai import *


net = MLPNeuralNetwork([2, 3, 2], activation=ActivationFunction)
net.layers[0] = [1, 1]

net.print()

net.forwardPropagation()

print()
net.print()

