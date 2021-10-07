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
        plt.close()
        plt.figure(figsize=(7,8))

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
        

        # Normalize input data
        self.train_inputs = self.normalize(self.train_inputs)

        # Read header data
        self.headers = headers[0].split(',')[0:-1]

        # Prepare / preprocess data
        self.n_train_samples = 2600
        self.prepare_data()

        # Inputs that failed training
        self.failures = []

        # Model parameters
        np.random.seed()
        self.weights = np.random.normal(0,1,self.train_inputs.shape[1])
        self.bias = np.random.normal(0,1,1)[0]

        # Model data
        self.errors = []            # List for keeping track of training progress
        self.curr_loss = 0.0            # Current loss
        self.trained = False

    # Preprocess normalized input data
    def prepare_data(self):

        # Create an array of values from 0-len(train_inputs_norm)
        indices = np.arange(self.train_inputs.shape[0])

        # Randomly shuffle array of indices
        indices = np.random.permutation(indices)

        # Create validation sets from last 2600 randomly-aranged, normalized inputs
        self.validation_inputs = self.train_inputs[indices[self.n_train_samples:]]
        self.validation_labels = self.train_labels[indices[self.n_train_samples:]]

        # Set training data as the rest of the normalized inputs
        self.train_inputs = self.train_inputs[indices[:self.n_train_samples]]
        self.train_labels = self.train_labels[indices[:self.n_train_samples]]


    # Logistic (sigmoid) function
    def sigmoid(self, x):
        # New array with the specified precision
        return 1 / (1 + np.exp(-x))

    # Regularize weights
    def L2(self):
        return np.linalg.norm(self.weights)

    # Normalize data
    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)

        return (data - mean) / std

    # Compute weighted sum of inputs
    def wsum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    # Compute weighted sum and activate (Feed forward)
    def feed_forward(self, inputs):
        return self.sigmoid(self.wsum(inputs))

    # Compute the cross entropy loss of a single prediction
    # boost = lambda = regularization hyperparameter
    def cross_entropy(self, predicted, actual, boost):
        
        if actual:
            return -np.mean(np.log(predicted))
        else:
            return -np.mean(np.log((1.0 - predicted) + (boost/2.0)*self.L2()))

    """
    Compute the gradient of the cross entropy loss function
    """
    def gradient(self, inputs, label, boost):
        return np.dot(inputs, (self.feed_forward(inputs) - label)) + (boost/2)*self.L2()

    # Make a prediction (either 0 or 1)
    def predict(self, inputs):
        return np.where(self.feed_forward(inputs) > 0.5, 1, 0)

    # Perform a weight update
    def update(self, inputs, label, learning_rate, boost):
        # Compute weight gradient
        gradient = self.gradient(inputs, label, boost)

        # Update weight vector by stepping in the direction of neg. gradient
        self.weights -= learning_rate * gradient

        # Update bias
        self.bias -=  learning_rate * (self.feed_forward(inputs) - label)


    # Train model on data
    def train(self, max_itrs=100, learning_rate=0.0001, boost=0.00001, stop=0.01, batch=''):
        
        # Initialize training results
        train_results = {
            'loss': [],
            'iterations': 0
        }

        # Start and stop points for training
        start = 0
        end = len(self.train_inputs)

        # Check if a batch has been specified
        if len(batch) == 2:
            start = int(batch[0])
            end = int(batch[1])

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


            # Check if input failed training
            if loss < 0:
                self.failures.append(start + i)

                

            # Plot losses, skip if curr_itr <= 1
            print("Loss(input=" + str(i) + "): " + str(loss) + "\titrs = " + str(curr_itr))
            

            # Update training results
            train_results['iterations'] += curr_itr

            if loss > 0:
                self.errors.append(1.0 - self.accuracy())


        
        # Plot learning progress
        plt_title = 'Training progress' + "\nMax Iterations = " + str(max_itrs) + "\nLearning Rate = " + str(learning_rate) + "  Boost = " + str(boost) + "\nCurrent Accuracy = " + str(self.accuracy() * 100.0) + "%"
        ModelVisualizer.plot_loss(self.errors, xlabel='Training iteration', ylabel='Loss (Error)', title=plt_title)
        plt.pause(2)

        
        print("\nMODEL TRAINED  (with %d failures)!\n" % (len(self.failures)))
        self.trained = True
        return train_results


    def accuracy(self):
        return np.sum((self.feed_forward(self.train_inputs)>0.5).astype(np.float64) == self.train_labels)  / self.train_labels.shape[0]


    # Save model to file
    def saveModel(self, filename):
        file = open(filename, 'w')

        # Store model parameters in file
        file.write("Weights: " + str(self.weights) + "\n\n")
        file.write("Bias: " + str(self.bias))

        file.close()

        # Check if save was successful
        if os.path.exists(filename):
            print("\n\nMODEL SAVED!\nModel file = " + str(filename))
        else:
            print("\n\nERROR in LogReg.saveModel(): Failed to save model!")


# Train/Test model on batch of data
def runBatch(b_start, b_end, itrs, learn_rate, boost, logReg, train=True):
    if train:   
        # Compute accuracy before training
        print("\nModel accuracy before training: %.3f%%\n" % (logReg.accuracy()*100.0))

        print("Training on batch [%d:%d] (size=%d)...\n" % (b_start, b_end, (b_start + b_end)))

        # Train model on batch data
        logReg.train(itrs, learn_rate, boost, batch=[b_start, b_end])

        # Compute accuracy after training
        print("\nModel accuracy after training: %.3f%%" % (logReg.accuracy()*100.0))

    else:
        raise NotImplementedError

def main():
    # Data files
    trainfile = 'datasets/spambase-train.csv'
    testfile = 'datasets/spambase-test.csv'

    # Hyperparameters
    max_train_itrs = 20000
    learning_rate = 0.001
    boost = 0.000001
    min_lr = 1e-6       # Minimum learning rate
    min_boost = 1e-9    # Minimum boost
    mod_rate = 2        # Number of batches to train before updating hyperparams

    # Model
    lr = LogReg(trainfile, testfile)

    # Batch data
    n_batches = 26 # 100 inputs per batch

    for i in range(n_batches):
        start = (len(lr.train_inputs) / n_batches) * i              # Start index of batch
        end = ((len(lr.train_inputs) / n_batches) * (i + 1)) - 1    # End index of batch

        # Adjust boost and learning rate every other batch as model trains (to avoid overfitting)
        if i % mod_rate == 0 and i > 0:
            if learning_rate > min_lr:
                # Last few updates
                if learning_rate <= min_lr*10:
                    max_train_itrs *= 0.7   # Update max training iterations
                
                mod_rate *= 2           # Change update rate
                learning_rate /= i  # update learning rate

            if boost > min_boost:
                boost /= i

        # Train model on batch
        runBatch(start, end, max_train_itrs, learning_rate, boost, lr)
    

    plt_title = 'Training progress' + "\nMax Iterations = " + str(max_train_itrs) + "\nLearning Rate = " + str(learning_rate) + "  Boost = " + str(boost) + "\nCurrent Accuracy = " + str(lr.accuracy() * 100.0) + "%"
    ModelVisualizer.plot_loss(lr.errors, xlabel='Training iteration', ylabel='Loss (Error)', title=plt_title)

    # Save model to file
    lr.saveModel('model.obj')
    plt.savefig('model_progress.png')


    

main()
    

