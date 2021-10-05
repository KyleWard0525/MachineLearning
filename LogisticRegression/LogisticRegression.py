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
import numpy as np

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
        self.learning_rate = 0.001

        # Model data
        self.errors = []                # List for keeping track of training progress
        self.curr_loss = 0.0            # Current loss
        self.trained = False

    # Logistic (sigmoid) function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of logistic function
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # Compute weighted sum of inputs
    def wsum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    # Compute weighted sum and activate (Feed forward)
    def feed_forward(self, inputs):
        return self.sigmoid(self.wsum(inputs))

    # Compute the cross entropy loss of a single prediction
    def cross_entropy(self, predicted, actual):
        # Check if predicted is correct
        if predicted == 0.0 or (1 - predicted) == 0.0:
            return 1.0

        return -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    """
    Compute the gradient of the cross entropy loss function
    """
    def gradient(self, inputs, label):

        # Derivative with respect to the model's prediction (output)
        loss_wrt_prediction = label * (1 / self.feed_forward(inputs)) + (1 - label) * 1 / (1 - self.feed_forward(inputs))

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
    def update(self, inputs, label):
        # Compute weight gradient
        gradient = self.gradient(inputs, label)

        # Update weight vector by stepping in the direction of neg. gradient
        self.weights += -self.learning_rate * gradient

        # Update bias
        self.bias += label


    # Compute total loss over the entire dataset
    def total_loss(self):
        loss = 0.0

        # Loop through dataset
        for i in range(len(self.train_inputs)):
            # Get input vector and class label
            input = self.train_inputs[i]
            label = self.train_labels[i]

            # Compute output
            output = self.feed_forward(input)
            
            # Increment loss 
            loss += abs(label - output)
        
        return loss


    # Train model on data
    def train(self, max_itrs=100, stuck_limit=5, stop=0.01):
        
        # Initialize training results
        train_results = {
            'loss': [],
            'iterations': 0
        }

        # Compute initial loss over entire training set
        init_loss = self.total_loss()
        print("\nStarting loss: " + str(init_loss))

        # Loop through dataset
        for i in range(len(self.train_inputs)):
            # Get input-label data
            inputs = self.train_inputs[i]
            label = self.train_labels[i]

            # Compute initial output and loss
            output = self.feed_forward(inputs)
            loss = self.cross_entropy(output, label)

            # Current training iteration for this input
            curr_itr = 0

            # Add total loss at iteration to training results
            #train_results['loss'].append(self.total_loss(train_inputs, train_labels))

            prev_loss = loss
            stuck_cnt = 0    # Keep track of number of times loss does not change

            # Train model on this input until it reaches acceptable margins for loss
            # or max iterations reached
            while loss > stop and curr_itr <= max_itrs:
                prev_loss = loss

                # Update model parameters
                self.update(inputs, label)

                # Recompute output and loss
                output = self.feed_forward(inputs)
                loss = self.cross_entropy(output, label)

                # Check if loss is the same (model is stuck)
                if loss == prev_loss:
                    stuck_cnt += 1
                if stuck_cnt >= stuck_limit:
                    print("MODEL STUCK IN LOCAL MINIMA! Moving to next input..\n")
                    break

                # Increment current training iterations
                curr_itr += 1

            if not np.isnan(loss):
                print("Loss(input=" + str(i) + "): " + str(loss))

                # Add total loss at iteration to training results
                #train_results['loss'].append(self.total_loss())

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

    # Test model on XOR dataset
    def testXOR(self):

        # Test results
        n_samples = len(self.test_inputs)
        n_correct = 0

        # Train model
        self.train(self.train_inputs, self.train_labels)

        print("\n\nTesting trained model on train data...\n")
        time.sleep(3)

        # Loop through training dataset
        for i in range(len(self.train_inputs)):
            # Get input-label data
            inputs = self.train_inputs[i]
            label = self.train_labels[i]

            prediction = self.predict(inputs)

            print("\nInputs: " + str(inputs))
            print("Predicted: " + str(prediction))
            print("Actual: " + str(label))

            time.sleep(2)

        print("\n\nTesting trained model on test data...\n")
        time.sleep(3)

        # Loop through test dataset
        for i in range(len(self.test_inputs)):
            # Get input-label data
            inputs = self.test_inputs[i]
            label = self.test_labels[i]

            # Make prediction
            prediction = self.predict(inputs)

            print("\nInputs: " + str(inputs))
            print("Predicted: " + str(prediction))
            print("Actual: " + str(label))

            # Check if prediction was correct
            if prediction == label:
                n_correct += 1

            time.sleep(2)

        print("\nTEST RESULTS:\n")
        print("Number of test samples: " + str(len(self.test_inputs)))
        print("Number of correct predictions: " + str(n_correct))
        print("\nModel Accuracy: %.2f%%" % ((n_correct / n_samples) * 100.0))



# Data files
trainfile = 'datasets/spambase-train.csv'
testfile = 'datasets/spambase-test.csv'

lr = LogReg(trainfile, testfile)

print("Length of inputs: " + str(len(lr.train_inputs)))

idx = 15
input = lr.train_inputs[idx]
label = lr.train_labels[idx]

print("\nTotal loss before training: " + str(lr.total_loss()))
lr.train(max_itrs=1000)
print("\nTotal loss before training: " + str(lr.total_loss()))


