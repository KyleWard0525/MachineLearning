"""
    Train an ML model on the MNIST dataset using tensorflow
"""
import tensorflow as tf
import numpy as np
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mlutils import MathML

# Check tf version
print("\nTensorflow version: " + str(tf.__version__))

# Create math object
math = MathML()

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

# Load train/test data from dataset
[train_inputs, train_labels], [test_inputs, test_labels] = mnist.load_data()

print("\nTraining size: " + str(len(train_inputs)))
print("Test size: " + str(len(test_labels)))
print("Number of inputs: " + str(len(train_inputs[0])))

# Normalize train and test inputs
train_inputs = math.normalize(train_inputs)
test_inputs = math.normalize(test_inputs)


# Create model (neural network)
model = tf.keras.models.Sequential([
    # Define network layers here

    # Input layer
    # 
    # Flatten the 28x28 image into a layer with 784 nodes. One for each pixel
    tf.keras.layers.Flatten(input_shape=[28,28]),   

    # First hidden layer 
    # 
    # fully-connected (dense), computes y = α(dot(inputs,weights) + bias)
    # 196 fully-connected nodes. ReLU activation function. L2 Weight Regularization. Adds bias to each node.
    tf.keras.layers.Dense(units=196, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0001), use_bias=True), 

    # Second hidden layer
    # 
    # Applies dropout to a random specified percentage of inputs
    # Applying dropout to an input simply means setting it eqaul to 0
    # This helps protect against overfitting 
    tf.keras.layers.Dropout(0.178), # Applies dropout to 20% of randomly selected inputs

    # Output layer 
    # 
    # Same as first hidden layer but with softmax activation (because we want a probability distribution of classes as our output)
    # It also only has 10 nodes as there are only 10 outputs. Thus, the prediction of the model will be the index of the element with
    # the highest probability
    tf.keras.layers.Dense(10, activation='softmax')
])

# Configure model training functions such as loss, optimizer algorighm, and accuracy metrics
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(x = train_inputs,                 #   Training inputs
          y = train_labels,                 #   Training labels
          batch_size = 1000,                #   Number of inputs to train on at once
          epochs = 50,                      #   Number of training iterations
          validation_split = 0.3,           #   Percentage of train data to be used by model as validation data at the end of each epoch
          verbose = 2,                      #   2 = print one line per epoch
          shuffle = True                    #   Shuffle training data at the beginning of each epoch
          )

# Evaluate the model on the training data
model.evaluate(test_inputs, test_labels)