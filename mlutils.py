"""
Basic utility functions for use in machine learning applications
kward
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# Math for machine learning
class MathML:


    # Initialize
    def __init__(self):
        
        self.activations = {
            'logistic': lambda x: 1 / (1 + np.exp(-x)),
            'linear': lambda x: 1 if x > 0.5 else 0,
            'relu': lambda x: 0 if x <= 0 else x,
            'logistic_deriv': lambda x: self.activations['logistic'](x)*(1-self.activations['logistic'](x)),
            'relu_deriv': lambda x: 0 if x < 0 else 1 
        }

        self.loss_functions = {
            'error': lambda predicted, actual: self.Loss.error(predicted,actual), 
            'mean_error': lambda outputs, labels: self.Loss.mean_error(outputs, labels),
            'mse': lambda outputs, labels: self.Loss.mean_sqr_error(outputs, labels),
            'bce': lambda outputs, labels: self.Loss.binary_cross_entropy(outputs, labels),
            'mse_deriv': lambda outputs, labels: self.Loss.mse_deriv(outputs, labels)
        }

    # L2 normalization
    def L2(self, data):
        return np.linalg.norm(data)

    # L_n normalization
    def Lnorm(self, data, n):
        return np.linalg.norm(data, ord=n)

    # Normalize data
    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)

        return (data - mean) / std

    # Compute weighted sum of inputs
    def wsum(self, inputs, weights, bias):
        return np.dot(inputs, weights) + bias

    
    #   Class for cost functions
    class Loss:
        # Regression Cost Functions 
        # (most basic loss function)
        @staticmethod
        def error(predicted, actual):
            return actual - predicted

        # Mean Error
        @staticmethod
        def mean_error(outputs, labels):
            return np.mean(np.sum(labels - outputs))

        # Squre error
        @staticmethod
        def sqr_error(predicted, label):
            return pow(label - predicted, 2)

        # Mean squared error
        @staticmethod
        def mean_sqr_error(outputs, labels):
            sum = 0.0 
            
            # Check labels type
            if isinstance(labels, int):
                #  Not an array. Return square error
                return pow(labels - outputs, 2)

            for i in range(len(labels)):
                sum += pow(labels[i] - outputs[i], 2)

            return sum / len(labels)

        # Mean square error derivative
        @staticmethod
        def mse_deriv(outputs, labels):
            # Check label type
            if isinstance(labels, int):
                return 2 * (labels - outputs)
                
            return np.mean(np.sum(2 * (labels - outputs)))

        # Mean absolute error
        @staticmethod
        def mean_abs_error(outputs, labels):
            return np.mean(np.sum(abs(labels - outputs)))

        # Binary classification cost function (NOTE: boost protects against: 1 - prediction = 0)
        @staticmethod
        def binary_cross_entropy(prediction, label, boost=0.000001):
            return -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction + boost))


    # Statistics inner class
    class Stats:
        def __init__(self):
            pass

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