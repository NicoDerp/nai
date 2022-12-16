#!/usr/bin/python3

from nai import *


net = MLPNeuralNetwork([2, 3, 2], activation=ReLU)
net.layers[0] = [0.1, 0.3]
net.expectedOutput = [0.8, 0.5]
net.learning_rate = 0.1

print("Trying to get:", net.expectedOutput)

max_epochs = 10000
epoch = 0

#for epoch in range(100):
while epoch < max_epochs and (err := net.globalError()) > 0.000001:
    print(f"Epoch {epoch+1}/{max_epochs}")

    net.forwardPropagation()
    net.backPropagation()
    print(f"\n{net}\n")
    print(f"Global error {err:.20f}")

    epoch += 1


