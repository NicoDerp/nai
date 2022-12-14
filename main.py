#!/usr/bin/python3


from nai import *


net = MLPNeuralNetwork([2, 3, 2], activation=ActivationFunction)
net.layers[0] = [1, 2]

print(f"\n{net}\n")

net.forwardPropagation()

print(f"\n{net}\n")

net.backPropagation()

print(f"\n{net}\n")

