#!/usr/bin/python3


from nai import *


net = MLPNeuralNetwork([2, 3, 2], activation=Sigmoid)
net.layers[0] = [1, 2]

net.forwardPropagation()
print(f"\n{net}\n")

print("Trying to get:", net.expectedOutput)

for epoch in range(10):
    print(f"Epoch {epoch+1}/10")

    net.forwardPropagation()
    print(f"\n{net}\n")

    net.backPropagation()

    net.forwardPropagation()
    print(f"\n{net}\n")



