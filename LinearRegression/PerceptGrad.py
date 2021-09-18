"""
Perceptron made to be optimized using gradient descent
"""

import numpy as np

# Perceptron class (loss function = squared error)
class GradientPerceptron:

    # Initialize perceptron
    def __init__(self, train_data):
        # Initialize class vars
        self.weights = np.zeros(len(train_data[0])-1)
        self.bias = 0.0
        self.data = train_data
        fl = self.split_features_labels()

        self.features = fl[0]
        self.labels = fl[1]
        

    # Split data into features (inputs) and labels (outputs)
    # Assuming label is last element in row
    def split_features_labels(self):
        features = []
        labels = []

        # Loop through row in data
        for row in self.data:
            labels.append(row[-1])                              # Extract label from last element in row
            features.append(row[0:len(row)-1])                  # Features are all other elements in row

            
        # Return features and labels
        return [features, labels]


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
            return 0     


    # Mean square error
    def mse(self):
        error = 0.0

        for row in self.data:
            output = self.activate(row[0:len(row)-1])               # Compute output for each row in train data
            label = row[-1]                                         # Extract label from train data

            error += pow(label - output, 2)

        return (1/len(self.data)) * error                           # Average the error
    
    # Compute gradient of a weight
    def gradient(self, weight_idx):
        grad = 0.0 
        weight = self.weights[weight_idx]

        for row in self.data:
            feats = row[0:len(row)-1]
            label = row[-1]

            grad += (-2 * feats[weight_idx]) * (label - (weight * feats[weight_idx] + self.bias))

        return (1/len(self.data)) * grad

    def update(self, lr):

        for weight_idx in range(len(self.weights)):
            self.weights[weight_idx] -= lr * self.gradient(weight_idx)

        return self.weights


    # Set model data
    def set_data(self, data):
        self.data = np.array(data)


def main():

    train_data = np.array([[1,0,0,1], [0,0,1,1], [0,0,0,0], [1,1,0,1], [1,0,1,1]]) #  XOR Data
    test_data = np.array([[0,1,1,1], [1,1,1,1], [0,1,0,1]])

    gp = GradientPerceptron(train_data)
    learn_rate = 0.01

    mse = gp.mse()
    print("\nmse = " + str(mse))


    print("\nWeights before update: " + str(gp.weights))
    gp.update(learn_rate)
    print("\nWeights after update: " + str(gp.weights))

    mse = gp.mse()
    print("\nmse = " + str(mse))


    for row in test_data:
        features = row[0:len(row)-1]
        label = row[-1]

        pred = gp.activate(features)

        print("\nFeatures: " + str(features))
        print("Label: " + str(label))
        print("Predicted: " + str(pred))


main()