"""
Multi-Layer Perceptron
"""
import os
import sys
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mlutils import MathML, split_inputs_labels

class MLP:

    def __init__(self, train_data='', n_hidden=0, activation='relu', loss_function='mse', regularizer=0.001):
        # Initialize class objects
        self.math = MathML()                                                            #   Machine Learning math "library"
        self.activate = self.math.activations[activation]                               #   Activation function
        self.activate_deriv = self.math.activations[activation + "_deriv"]              #   Derivative activation function
        self.loss_func = self.math.loss_functions[loss_function]                        #   Loss function
        self.loss_deriv = self.math.loss_functions[loss_function + "_deriv"]            #   Derivative loss function
        self.regularizer = regularizer                                                  #   Weight regularization hyperparameter

        # Split training data into inputs and labels
        self.train_data = split_inputs_labels(train_data)

        print("\nTrain Data 0: " + str(self.train_data['inputs'][0]) + "\tTrain Label 0: " + str(self.train_data['labels'][0]))

        # Normalize inputs
        self.train_data['inputs'] = self.math.normalize(self.train_data['inputs'])

        # Set number of inputs, hidden nodes, and outputs
        self.n_inputs = len(self.train_data['inputs'][0])
        self.n_hidden = n_hidden

        if isinstance(self.train_data['labels'][0], int):
            self.n_outputs = 1
        else:
            self.n_outputs = len(self.train_data['labels'][0])
        

        # Model parameters
        np.random.seed()
        self.hidden_weights = [np.random.normal(0,1,self.n_inputs) for i in range(self.n_hidden)]      # Input to Hidden layer weights
        self.output_weights = np.random.normal(0,1,self.n_outputs * self.n_hidden)         # Hidden to Output layer weights
        self.biases = np.random.normal(0,1,n_hidden)                                       # Biases for hidden neurons

        print("Loss: " + str(self.loss_func([1,0], [1,1])))


    # Propagate data through the network
    def feed_forward(self):
        
        """""""""""""""""""""""""""""""""""
                
                    HIDDEN LAYER

        """""""""""""""""""""""""""""""""""
        # Values of hidden nodes (wsum of inputs and hidden weights)
        hidden_vals = []

        # Loop through inputs 
        for i in range(len(self.hidden_weights)):
            hidden_vals.append(self.math.wsum(self.train_data['inputs'][0], self.hidden_weights[i], self.biases[i]))
        
        print("\nValues of hidden nodes after partial propagation: " + str(hidden_vals))
        
        # Activate hidden layer
        for i in range(len(hidden_vals)):
            hidden_vals[i] = self.activate(hidden_vals[i])
        
        print("Values of hidden nodes after activation: " + str(hidden_vals))

        
        """""""""""""""""""""""""""""""""""
                
                    OUTPUT LAYER

        """""""""""""""""""""""""""""""""""
        # Compute unactivated output
        output = self.math.wsum(hidden_vals, self.output_weights, 0) # 0 because output node has no bias

        print("\nValue of output node before activation: " + str(output))

        # Activate output
        output = self.math.activations['linear'](output)

        print("Value of output node after activation: " + str(output))

        return output


    # Compute loss with weight regularization
    def cost(self, prediction, label):
        return self.loss_func(prediction, label) + ()



    # Print information about MLP
    def print_self(self):
        print("\n\n--------------   MLP  ---------------\n\n")
        print("Training samples: " + str(len(self.train_data['inputs'])))
        print("\nInput layer size: " + str(len(self.train_data['inputs'][0])))
        print("Hidden layer size: " + str(self.n_hidden))
        print("Output layer size: " + str(self.n_outputs))
        print("\nHidden Layer Weights: " + str(self.hidden_weights))
        print("Output Layer Weights: " + str(self.output_weights))
        print("\nBiases for Hidden nodes: " + str(self.biases))
        print("\n\nInputs: " + str(self.train_data['inputs']))
        print("\nLabels: " + str(self.train_data['labels']))
        print("\n\n----------------------------------------------\n\n")

