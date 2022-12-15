#!/usr/bin/python3


from nai import *


net = MLPNeuralNetwork([2, 3, 2], activation=Sigmoid)
net.layers[0] = [1, 2]

net.forwardPropagation()
print(f"\n{net}\n")

for epoch in range(10):
    print(f"Epoch {epoch+1}/10")

    print("Trying to get:", net.expectedOutput)

    print(net.weights)

    net.forwardPropagation()
    net.backPropagation()
    print(f"\n{net}\n")
    print("Global error", net.globalError())

#    net.forwardPropagation()
#    print(f"\n{net}\n")



