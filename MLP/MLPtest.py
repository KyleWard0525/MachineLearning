"""
Test harness for MLP.py
"""

from MLP import MLP
import os
import sys
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mlutils import MathML, split_inputs_labels
from datagen import DataGen


class MLPTest:

    def __init__(self):
        dg = DataGen()              # Generates dataset(s)
        self.math = MathML()        # Math for machine learning
        n_samples = 35

        # Initialize datasets
        self.raw_dataset = dg.bmi_dataset(n_samples)    
        
        mlp = MLP(self.raw_dataset, n_hidden=3)

        mlp.print_self()
        prediction = mlp.feed_forward()
        label = mlp.train_data['labels'][0]
        cost = mlp.cost(prediction, label)
        print("\n\nModel's initial prediction on first sample:")
        print("Prediction: " + str(prediction) + "\tLabel: " + str(label))
        print("Cost: " + str(cost) + "\n")

        mlp.gradient(prediction, label)



tester = MLPTest()
