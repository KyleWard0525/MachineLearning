"""
This file is an attempt to train a linear regression classifier to predict if
an individual is overweight given their height and weight.

Activation function = ReLU
Loss function = Mean squared error

"""

import os
import sys
import numpy as np

# Add parent directory to system path to import mlutils
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mlutils import MathML, split_inputs_labels
from datagen import DataGen

# BMI Classifier using linear regression
class BMI_Classifier:

    # Initialize model
    def __init__(self, n_samples):
        dg = DataGen()              # Generates dataset(s)
        self.math = MathML()        # Math for machine learning

        # Initialize datasets
        self.raw_dataset = dg.bmi_dataset(n_samples)    
        self.split_dataset = split_inputs_labels(self.raw_dataset)

        # Initialize model parameters
        self.weights = np.ones(len(self.split_dataset['inputs'][0]))
        self.bias = np.random.uniform(0,1,1)

        # Normalize inputs 
        self.split_dataset['inputs'] = self.math.normalize(self.split_dataset['inputs'])

        print("\nWeights: " + str(self.weights))
        print("\nBias: " + str(self.bias))
        print("\nInputs: " + str(self.split_dataset['inputs']))
        print("\nLabels: " + str(self.split_dataset['labels']))

    # Propagate forward and compute outputs
    def make_predictions(self):
        predictions = []

        # Loop through inputs
        for row in self.split_dataset['inputs']:
            # Compute weighted sum of input vector
            wsum = self.math.wsum(row, self.weights, self.bias)

            # Compute prediction and add to list
            predictions.append(self.math.activations['relu'](wsum))

        return predictions

    # Compute total loss
    def loss(self):
        return self.math.Loss.mean_sqr_error(self.make_predictions(), self.split_dataset['labels'])

    # Compute accuracy
    def accuracy(self):
        # Get predictions
        preds = self.make_predictions()
        
        # Number of correct predictions
        n_correct = 0

        # If prediction is ndarray, cast to float
        for pred in preds:
            if isinstance(pred, np.ndarray):
                pred = pred[0]

        print("\nPredictions: " + str(preds))

        # Loop through labels
        for i in range(len(self.split_dataset['labels'])):
            label = self.split_dataset['labels'][i]

            pred = preds[i]

            if preds[i] == label:
                n_correct += 1

        return float(len(preds)/n_correct)


bmic = BMI_Classifier(37)
print("\nInitial accuracy: %.2f%%" % bmic.accuracy())
print("\nInitial loss: " + str(bmic.loss()))

