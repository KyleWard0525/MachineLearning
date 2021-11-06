import os, sys
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Imager import Imager
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
print("Input Shape: [" + str(train_inputs.shape[1]) + ", " + str(train_inputs.shape[2]) + "]")

# Normalize inputs
train_inputs = train_inputs / 255.0
test_inputs = test_inputs / 255.0

mlp_train_inputs = train_inputs.reshape(-1, train_inputs.shape[1]*train_inputs.shape[2])
mlp_test_inputs = test_inputs.reshape(-1, test_inputs.shape[1]*test_inputs.shape[2])

# Define hyperparameters for both models
hidden_nodes = 256                  #   Number of nodes in the hidden layer
epochs = 10                         #   Training iterations
steps = 100                         #   Training steps per iteration
batch_size = 100                    #   Number of samples to train on at once
valid_split = 0.3                   #   % of train data to be set aside for validation during training
verbosity = 2                       #   Print 1 line per epoch
shfl = True                         #   Shuffle training data at the beginning of each epoch
n_workers = 6                       #   Maximum of 6 parallel PROCESSES
multiproc = True                    #   Use multi-processing while training



# Create MLP model structure
mlp_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=mlp_train_inputs.shape[1:2]),       #   Input layer, flatten image into 1D vector
    tf.keras.layers.Dense(units=hidden_nodes, activation='relu'),       #   Hidden layer, 128 neurons, relu activation
    tf.keras.layers.Dense(units=10, activation='softmax')               #   Output layer, 10 outputs, 1 for each digit 1-10
])

# Configure MLP model training
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp_model._name = "MLP_MNIST_MODEL"
# Print model information
print("\n" + str(mlp_model.summary()))

# Train and test model
mlp_model.fit(
    x = mlp_train_inputs, 
    y = train_labels,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbosity,
    validation_split=valid_split,
    shuffle=shfl,
    steps_per_epoch=steps,
    validation_steps=steps,
    validation_batch_size=batch_size,
    workers=n_workers,
    use_multiprocessing=multiproc
    )

# Evaluate and save model
mlp_test_loss, mlp_test_acc = mlp_model.evaluate(test_inputs, test_labels)

print("\nMLP Loss: " + str(mlp_test_loss))
print("MLP Accuracy: " + str(mlp_test_acc))

tf.keras.models.save_model(mlp_model, 'models/mlp_mnist')



# Reshape inputs for use in CNN
train_inputs = train_inputs.reshape(train_inputs.shape[0], train_inputs.shape[1], train_inputs.shape[2], 1)
test_inputs = test_inputs.reshape(test_inputs.shape[0], test_inputs.shape[1], test_inputs.shape[2], 1)



# Create CNN model
cnn_model = tf.keras.models.Sequential([
    # Apply 64 3x3 convolutional filters to each image and activate using ReLU
    tf.keras.layers.Conv2D(64, [3,3], activation='relu', input_shape=(train_inputs.shape[1], train_inputs.shape[2], 1)),

    # Apply 2x2 Max-Pooling. Take each 2x2 block of the image and pick the maximum value in that block
    # This help reduces image size and complexity without losing features
    tf.keras.layers.MaxPooling2D(2, 2),

    # Apply the normal MLP network to the now convolved and pooled inputs
    tf.keras.layers.Flatten(input_shape=train_inputs.shape[1:2]),       #   Input layer, flatten image into 1D vector
    tf.keras.layers.Dense(units=hidden_nodes, activation='relu'),       #   Hidden layer, 128 neurons, relu activation
    tf.keras.layers.Dense(units=10, activation='softmax')               #   Output layer, 10 outputs, 1 for each digit 1-10
])

# Configure CNN model training
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model._name = "CNN_MNIST_MODEL"
# Print model information
print("\n" + str(cnn_model.summary()))

# Train and test model
cnn_model.fit(
    x = train_inputs, 
    y = train_labels,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbosity,
    validation_split=valid_split,
    shuffle=shfl,
    steps_per_epoch=steps,
    validation_steps=steps,
    validation_batch_size=batch_size,
    workers=n_workers,
    use_multiprocessing=multiproc
    )


# Evaluate and save model
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_inputs, test_labels)

print("\n\nCNN Loss: " + str(cnn_test_loss))
print("CNN Accuracy: " + str(cnn_test_acc))

tf.keras.models.save_model(cnn_model, 'models/cnn_mnist')


# Compare models
dAcc = ((cnn_test_acc - mlp_test_acc) / abs(cnn_test_acc)) * 100.0

print("\n\nMODEL RESULTS:\n\n")
print("CNN improved its accuracy by " + str(dAcc) + "%")