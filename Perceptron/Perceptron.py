"""
A basic Perceptron for Machine Learning applications
"""
import numpy as np
import time

class Perceptron:

    #   Initialize perceptron
    def __init__(self):
        # Initialize variables
        self.bias = 0.0
        self.trained = False

    #   Compute weighted sum of data vector
    def compute_sum(self, data):
        weighted_sum = 0.0

        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            # Force data to be a numpy array
            data = np.array(data)

        # Loop through data and compute weighted sum
        for i in range(len(data)):
            weighted_sum += data[i] * self.weights[i]

        # Add bias to sum
        weighted_sum += self.bias

        return weighted_sum

    #   Compute model activation
    def activate(self, data):
        sum = self.compute_sum(data)

        # Activate using the sign of the sum (sum<0=-1, sum>0=1)
        if sum > 0:
            return 1
        else:
            return -1

    #   Train model on dataset
    def train(self, train_data, numEpochs):
        self.weights = np.zeros(len(train_data))

        # Train for n Epochs
        for i in range(numEpochs):
            # Loop through all input-output pairs
            for row in train_data:
                # Extract inputs and label from row
                label = row[-1]

                # Convert label if needed
                if label < 1:
                    label = -1

                inputs = row[0:len(row)-1]

                # Compute activation
                activation = self.activate(inputs)

                print("Inputs = " + str(inputs) + ", Label = " + str(label)
                + ", activation = " + str(activation))

                # Check for error
                if activation != label:
                    # Error found, update weights and bias
                    for weight in range(len(inputs)):
                        # Compute update for new weights
                        self.weights[weight] += (label * inputs[weight])

                    # Compute update for new bias
                    self.bias += label

                    # Print updates
                    print("\nError found: Updating weights and bias..\nNew weights: " + str(self.weights) + "\nNew bias: " + str(self.bias) + "\n")

        # Signal that training is complete
        self.trained = True
                
    #   Make a prediction using the trained model
    def predict(self, inputs):
        # Run inputs through model
        self.train(inputs, 2)

        # Compute prediction
        predictions = np.zeros(len(inputs))

        for i in range(len(inputs)):
            # Compute predictions (skip training label)
            input = inputs[i][0:len(inputs[i])-1]
            print("input=" + str(input))
            predictions[i] = self.activate(input)

        print("Predicitons: " + str(predictions))

        return predictions





train_data = np.array([[1,0,0,1], [0,0,1,1], [0,0,0,0], [1,1,0,1], [1,0,1,1]]) #  XOR Data

test_data = np.array([[0,1,1,1], [1,1,1,1], [0,1,0,1]])

p = Perceptron()

p.train(train_data, 3)

print("\nModel trained!\nMaking prediction on test_data...")
time.sleep(1)

test_labels = np.zeros(len(test_data))

for i in range(len(test_data)):
    test_labels[i] = test_data[i][-1]

print("\nTEST PARAMETERS:\n")
print("Inputs: " + str(test_data))
print("Labels: " + str(test_labels))

print("\nMODEL PARAMETERS:\n")
print("Weights: " + str(p.weights))
print("Bias: " + str(p.bias))

output = p.predict(test_data) #  Exclude label

print("\n\nTEST RESULTS:\n")
print("Predictions: " + str(output))
print("Labels: " + str(test_labels))
