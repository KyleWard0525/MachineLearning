"""
Basic utility functions for use in machine learning applications

kward
"""

import numpy as np

# Split data into inputs and label (assume label is last element in row)
def split_inputs_labels(data):
    inputs = []
    labels = []

    # Loop through data
    for row in data:

        # Add row label to list
        labels.append(row[-1])

        # Add remaining elements to inputs
        inputs.append(row[0:len(row)-1])

    # Return results as a dict
    return {'inputs': inputs,
            'labels': labels}

# Compute weighted sum of inputs
def wsum(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

# Logistic (sigmoid) function
def sigmoid( x):
    return 1 / (1 + np.exp(-x))

# Derivative of logistic function
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))