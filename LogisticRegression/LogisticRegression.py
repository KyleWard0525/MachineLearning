"""
3rd attempt at implementing logistic regression
with cross entropy loss and gradient descent

kward
"""
import sys
import os
import time
import re
import warnings
import random
import numpy as np
from functools import cache
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import mlutils


# Global variables (FOR TESTING PURPOSES)
train_data = np.array([[1,0,0,1], [0,0,1,1], [0,0,0,0], [1,1,0,1], [1,0,1,1]]) #  XOR Data
test_data = [[0,1,1,1], [1,1,1,1], [0,1,0,1]]

train_set = mlutils.split_inputs_labels(train_data)
test_set = mlutils.split_inputs_labels(test_data)

headers = []

class Parser:

    # Read in dataset (code from logistic-regression.ipynb)
    @staticmethod
    def read_data(filename):
        global headers 

        f = open(filename, 'r')
        p = re.compile(',')
        xdata = []
        ydata = []
        header = f.readline().strip()
        headers.append(header)
        varnames = p.split(header)
        namehash = {}
        for l in f:
            li = p.split(l.strip())
            xdata.append([float(x) for x in li[:-1]])
            ydata.append(float(li[-1]))
        
        return np.array(xdata), np.array(ydata)

# Model Visualizer class
class ModelVisualizer:

    # Initialize
    def __init__(self):
        pass

    # Plot losses over iterations
    @staticmethod
    def plot_loss(losses, xlabel='', ylabel='', title=''):
        # Create a linear space to plot
        x_axis = np.linspace(0,len(losses), len(losses))

        # Plot losses and show figure
        plt.plot(x_axis,losses, 'r')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.autoscale(True,axis='y')
        plt.show(block=False)

# Logistic Regression Class
class LogReg:

    # Initialize class object
    def __init__(self, train_file='', test_file=''):

        # QOL Checks for using default datasets
        if train_file and not test_file:
            # Read inputs and labels from datasets
            self.train_inputs, self.train_labels = Parser.read_data(train_file)
            warnings.warn("\nWARNING in LogReg.__init__(): train_file given without test_file!")

        elif test_file and not train_file:
            print("\nERROR in LogReg.__init__(): test_file given without a train_file! Falling back to default XOR dataset")
            
            # Read default datasets
            self.train_inputs = train_set['inputs']
            self.train_labels = train_set['labels']

            # Testing data
            self.test_inputs = test_set['inputs']
            self.test_labels = test_set['labels']

        elif test_file and train_file:
            # Read inputs and labels from datasets
            self.train_inputs, self.train_labels = Parser.read_data(train_file)
            self.test_inputs, self.test_labels = Parser.read_data(test_file)

        else:
            warnings.warn("\nWARNING in LogReg.__init__(): No data given! Falling back to default XOR dataset")

            # Read default datasets
            self.train_inputs = train_set['inputs']
            self.train_labels = train_set['labels']

            # Testing data
            self.test_inputs = test_set['inputs']
            self.test_labels = test_set['labels']
        

        # Model parameters
        self.weights = np.random.rand(len(self.train_inputs[0]))
        self.bias = 0.0

        # Model data
        self.errors = []                # List for keeping track of training progress
        self.curr_loss = 0.0            # Current loss
        self.trained = False

    # Logistic (sigmoid) function
    @cache
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of logistic function
    @cache
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # Regularize weights
    def L2(self):
        return np.linalg.norm(self.weights - self.bias)

    # Compute weighted sum of inputs
    def wsum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    # Compute weighted sum and activate (Feed forward)
    def feed_forward(self, inputs):
        return self.sigmoid(self.wsum(inputs))

    # Compute the cross entropy loss of a single prediction
    # boost = lambda = regularization hyperparameter
    def cross_entropy(self, predicted, actual, boost):
        return -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted + (boost/2)*self.L2()))

    """
    Compute the gradient of the cross entropy loss function
    """
    def gradient(self, inputs, label, boost):

        # Derivative with respect to the model's prediction (output)
        loss_wrt_prediction = label * (1 / self.feed_forward(inputs)) + (1 - label) * 1 / (1 - self.feed_forward(inputs) + (boost/2)*self.L2())

        # Derivative of output with respect to it's weighted sum
        pred_wrt_wsum = self.sigmoid_deriv(self.wsum(inputs))

        # Derivative of weighted sum with respect to weights
        wsum_wrt_weights = inputs

        # Compute and return gradient
        gradient = loss_wrt_prediction * pred_wrt_wsum * wsum_wrt_weights
        return gradient

    # Make a prediction (either 0 or 1)
    def predict(self, inputs):
        return round(self.feed_forward(inputs))

    # Perform a weight update
    def update(self, inputs, label, learning_rate, boost):
        # Compute weight gradient
        gradient = self.gradient(inputs, label, boost)

        # Update weight vector by stepping in the direction of neg. gradient
        self.weights += learning_rate * gradient - boost

        # Update bias
        self.bias += label


    # Compute total loss over the entire dataset
    def total_loss(self, boost, inputs='', labels=''):
        loss = 0.0

        # Compute total loss of existing training set
        if len(inputs) <= 0 and len(labels) <= 0:
            # Loop through dataset
            for i in range(len(self.train_inputs)):
                # Get input vector and class label
                input = self.train_inputs[i]
                label = self.train_labels[i]

                # Compute output
                output = self.feed_forward(input)
                
                # Increment loss 
                loss += self.cross_entropy(output, label, boost)

        # Compute total loss of given data    
        else:
            # Loop through data
            for i in range(len(inputs)):
                # Get input vector and class label
                input = inputs[i]
                label = labels[i]

                # Compute output
                output = self.feed_forward(input)
                
                # Increment loss 
                loss += self.cross_entropy(output, label, boost)

            
        return loss


    # Train model on data
    def train(self, max_itrs=100, learning_rate=0.001, boost=0.00001, stop=0.01, batch=''):
        
        # Initialize training results
        train_results = {
            'loss': [],
            'iterations': 0
        }

        # Inputs that failed and need to be re-trained after normal training
        failures = []

        # Compute initial loss over entire training set
        init_loss = self.total_loss(boost)

        # Start and stop points for training
        start = 0
        end = len(self.train_inputs)

        # Check if a batch has been specified
        if len(batch) == 2:
            start = batch[0]
            end = batch[1]

        # Loop through dataset
        for i in range(start, end):
            # Get input-label data
            inputs = self.train_inputs[i]
            label = self.train_labels[i]

            # Compute initial output and loss
            output = self.feed_forward(inputs)
            loss = self.cross_entropy(output, label, boost)

            # Current training iteration for this input
            curr_itr = 0

            # Loss values for the current input
            input_losses = []
            input_losses.append(loss)

            # Add total loss at iteration to training results
            #train_results['loss'].append(self.total_loss(train_inputs, train_labels))

            # Train model on this input until it reaches acceptable margins for loss
            # or max iterations reached
            while loss > stop and curr_itr < max_itrs:

                # Update model parameters
                self.update(inputs, label, learning_rate, boost)

                # Recompute output and loss
                output = self.feed_forward(inputs)
                loss = self.cross_entropy(output, label, boost)

                # Increment current training iterations
                curr_itr += 1

                # Add current loss at current iteration to list
                input_losses.append(loss)

                # Add total loss at iteration to training results
                #train_results['loss'].append(self.total_loss())


            # Check if input failed training
            if loss > stop:
                failures.append(i)

            # Plot losses, skip if curr_itr <= 1
            print("Loss(input=" + str(i) + "): " + str(loss) + "\titrs = " + str(curr_itr))
            plt_title = 'Training progress on input ' + str(i)

            # if curr_itr > 1:
            #     ModelVisualizer.plot_loss(input_losses, xlabel='Training iteration', ylabel='Loss (Error)', title=plt_title)
            #     plt.pause(1)
            #     plt.close()

            # Increment total training iterations
            train_results['iterations'] += curr_itr

        
        print("\nMODEL TRAINED!\n")
        self.trained = True
        return train_results

    # Load train/test data from file
    def loadDataFromFile(self, filename, train_data=True):
        # Check if training data or testing data
        if train_data:
            self.train_inputs, self.train_labels = Parser.read_data(filename)
            print("\nNew training data loaded!\n")
        else:
            self.test_inputs, self.test_labels = Parser.read_data(filename)
            print("\nNew test data loaded!\n")


# Train/Test model on batch of data
def runBatch(b_start, b_end, itrs, learn_rate, boost, logReg, train=True):
    if train:
        # Get batch data from logreg obj
        input_data = logReg.train_inputs[b_start:b_end]
        input_labels = logReg.train_labels[b_start:b_end]

        print("Training on batch [%d:%d] (size=%d)...\n" % (b_start, b_end, (b_start + b_end)))
        print("Total loss before training: " + str(logReg.total_loss(boost, inputs=input_data, labels=input_labels)))

        # Train model on batch data
        logReg.train(itrs, learn_rate, boost, batch=[b_start, b_end])

        print("Total loss after training: " + str(logReg.total_loss(boost, inputs=input_data, labels=input_labels)))

    else:
        raise NotImplementedError

def main():
    # Data files
    trainfile = 'datasets/spambase-train.csv'
    testfile = 'datasets/spambase-test.csv'

    # Hyperparameters
    max_train_itrs = 100000
    learning_rate = 0.01
    boost = 0.00001

    # Model
    lr = LogReg(trainfile, testfile)

    # Batch data
    batch_start = 0
    batch_end = 100

    # Train on batch
    runBatch(batch_start, batch_end, max_train_itrs, learning_rate, boost, lr)
    

main()
    

