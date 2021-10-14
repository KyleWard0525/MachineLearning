"""
Basic utility functions for use in machine learning applications

kward
"""

import numpy as np
import matplotlib.pyplot as plt

# Math for machine learning
class MathML:


    # Initialize
    def __init__(self):
        
        self.activations = {
            'logistic': lambda x: 1 / (1 + np.exp(-x)),
            'identity': lambda pred, label: 1 if pred == label else 0,
            'relu': lambda x: 0 if x <= 0 else x,
            'logistic-deriv': lambda x: self.activations['logistic'](x)*(1-self.activations['logistic'](x)),
            'relu-deriv': lambda x: 0 if x < 0 else 1 
        }

    # L2 normalization
    def L2(self, data):
        return np.linalg.norm(data)

    # Lₙ normalization
    def Lnorm(self, data, n):
        return np.linalg.norm(data, ord=n)

    # Compute weighted sum of inputs
    def wsum(inputs, weights, bias):
        return np.dot(inputs, weights) + bias



# Data generator
class DataGen:

    def __init__(self):
        self.fig = plt.figure()

        self.conversions = {
            'in2cm': 2.54,
            'lb2kg': 0.453592
        }

    # Generate a dataset to classify overweight patients based on height and weight
    def bmi_dataset(self, n_samples):
        dataset = {
            'pos': {'weights': [], 'heights': []},  # Positive for overweight patients
            'neg': {'weights': [], 'heights': []}   # Negative for healthy patients
            }

        max_weight = 250
        min_weight = 50
        max_height = 80
        min_height = 36

        ppl_weights = np.random.uniform(low=min_weight, high=max_weight, size=n_samples)             # Weight in lbs
        ppl_heights = np.random.uniform(low=min_height, high=max_height, size=n_samples)              # Height in inches

        metric_weights = ppl_weights * self.conversions['lb2kg']  # Weight in kg
        metric_heights = ppl_heights * self.conversions['in2cm']  # Height in cm

        # Loop through people and compute their BMI
        for i in range(n_samples):
            # Compute bmi
            bmi = (metric_weights[i] / metric_heights[i] / metric_heights[i]) * 10000.00

            # Check if patient is overweight
            if bmi > 24.9:
                dataset['pos']['weights'].append(ppl_weights[i])
                dataset['pos']['heights'].append(ppl_heights[i])
            else:
                dataset['neg']['weights'].append(ppl_weights[i])
                dataset['neg']['heights'].append(ppl_heights[i])
       

        bmi_plot = self.fig.add_subplot(111)

        # Plot positive points
        bmi_plot.scatter(dataset['pos']['weights'], dataset['pos']['heights'], c="red", marker="+", label="Overweight (BMI > 24.9)")
        bmi_plot.scatter(dataset['neg']['weights'], dataset['neg']['heights'], c="blue", marker="_", label="Not Overweight")
        bmi_plot.set_title("BMI Plot")
        bmi_plot.set_xlabel("Weight(lbs)")
        bmi_plot.set_ylabel("Height(in)")
        bmi_plot.legend(loc='best', bbox_to_anchor=(1.05,1.15))
        plt.show()






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




data = np.random.rand(10)
weights = np.random.rand(len(data))
bias = np.random.normal(0,1)
math = MathML()


dg = DataGen()
dg.bmi_dataset(100)