# neural network using primarily OOP
from layers import Dense, ReLU, Tanh, Softmax, Sigmoid
from loss_functions import mse, mse_prime, bce, bce_prime
import numpy as np
from helpers.small.get_data import *
from helpers.small.network_evaluation import *
from grade_conversion import font_to_num, num_to_font
import tkinter as tk
import pickle

def train_network(X, Y, epochs, learning_rate, network):
    # loop to adjust network many times (gradient descent)
    for epoch in range(epochs):
        batches = 10
        lower = int((epoch % batches) * (len(X) / batches))
        upper = int((epoch % batches) * (len(X) / batches) + (len(X) / batches))
        #print(lower, upper)
        x_batch, y_batch = X[lower:upper], Y[lower:upper]
        error = 0
        # loop to train across all data
        for x, y in zip(x_batch, y_batch):
            # forward prop; x is first "output" since it is what comes out of the input layer
            output = x
            for layer in network:
                output = layer.forward(output)
            # keep running sum of error so average can be calculated
            #error += mse(y, output)
            error += bce(y, output)
            
            # backward prop; first gradient is gradient of error function
            #gradient = mse_prime(y, output)
            gradient = bce_prime(y, output)
            for layer in reversed(network):
                # layer.to_string()
                gradient = layer.backward(gradient, learning_rate)
        # to get average error
        error /= len(X)
        print(f"Epoch: {epoch}\tError: {error}")
    return network

def main():
    # getting data
    climbs = open_file()
    # training data and raw grades for evaluating network concisely
    x_data, y_data, raw_grades = parse_file(climbs, False)
    # initializing network options and layers
    epochs = 100
    learning_rate = 0.01
    network = [
        Dense(198, 50),
        Tanh(),
        Dense(50, 16),
        Softmax()
    ]
    # training network
    trained_network = train_network(x_data, y_data, epochs, learning_rate, network)
    with open("network.pkl", "wb") as file:
        pickle.dump(trained_network, file)
    
main()
