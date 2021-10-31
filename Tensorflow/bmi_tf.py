"""
This file is an attempt at using tensorflow to train an MLP to
classify whether or not someone is overwieght given their height and 
weight
"""
import os, sys
from numpy.linalg.linalg import norm
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

# Add parent directory for external modules
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mlutils import MathML, split_inputs_labels
from datagen import DataGen

# Check tf version
print("\nTensorflow loaded successfully.\tVersion: " + str(tf.__version__) + "\n")

# Preprocess data
def preprocess(dataset, test_amt, normalize_inputs=True):
    print("\nPreprocessing data...\n")

    # Processed data
    proc_data = {
        'train': {
            'inputs': [],
            'labels': []
        },
        'test': {
            'inputs': [],
            'labels': []
        }
    }
    # Split data into inputs and labels
    split_data = split_inputs_labels(dataset)

    # Split into training and testing sets
    proc_data['train']['inputs'], proc_data['test']['inputs'], proc_data['train']['labels'], proc_data['test']['labels'] = train_test_split(
                                                            split_data['inputs'], split_data['labels'], test_size=test_amt)


    # Cast labels to numpy arrays
    proc_data['train']['labels'] = np.array(proc_data['train']['labels'])
    proc_data['test']['labels'] = np.array(proc_data['test']['labels'])


    # Normalize data if specified
    if normalize_inputs:
        proc_data['train']['inputs'] = math.normalize(proc_data['train']['inputs'])
        proc_data['test']['inputs'] = math.normalize(proc_data['test']['inputs'])


    return proc_data






# Create generator and generate bmi dataset
datgen = DataGen()
n_samples = 100000
dataset = datgen.bmi_dataset(n_samples)


# Create math ML object
math = MathML()

# Define the model's hyperparameters
epochs = 10                    #   Training iteration
n_inputs = 2                    #   2 input nodes. 1 for weight, 1 for height
n_hidden = 4                    #   Hidden nodes
n_outputs = 1                   #   1 output (0 = not overweight, 1 = overweight)
train_amt = 0.8                 #   70% of data goes to training
train_valid_amt = 0.33           #   25% of train data is reserved for validation set
test_amt = 1.0 - train_amt      #   Rest of data goes to testing
hidden_activation = 'relu'      #   Hidden layer activation function
output_activation = 'sigmoid'    #   Output layer activation function
learn_rate = 0.001              #   Learning rate
weight_reg = 0.005           #   Weight regularization parameter
optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)                  #   Optimization algorithm
loss = 'mse'                                   #   Binary Cross Entropy loss function
batch_size = 128                #   Number of training samples to train at once
verbosity = 2                   #   2 = print line per epoch


#   Preprocess data to use in the model
model_data = preprocess(dataset, test_amt, normalize_inputs=True)

# Build and train model
def build_model():

    """     
                Create the model    

        Structure:

            Input layer: Input, 2 nodes
            Hidden layer 1: Dense, 3 node, activation=relu
            Output layer: Dense, 1 node, activation=linear
    """
    model = tf.keras.models.Sequential([

        #   Input layer
        tf.keras.layers.InputLayer(input_shape=(n_inputs,)),

        #   Hidden layer 1
        tf.keras.layers.Dense(units=n_hidden, kernel_regularizer=tf.keras.regularizers.L2(weight_reg), use_bias=True),

        #   Output layer
        tf.keras.layers.Dense(n_outputs, activation=output_activation)
    ])

    # Configure model training functions such as loss, optimizer algorighm, and accuracy metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model
    model.fit(
        x = model_data['train']['inputs'],                      #   Training inputs
        y = model_data['train']['labels'],                      #   Training labels
        steps_per_epoch = 100,                                  #   Number of updates to make per input per epoch
        epochs = epochs,                                        #   Number of training iterations
        verbose = verbosity,                                    #   Print out progress
        shuffle = True,                                         #   Shuffle train data every epoch
        batch_size = batch_size,                                #   Set batch size
        validation_split = train_valid_amt                      #   Amount of train data to set aside for validation
    )

    print("\n\nEvaluating model on test set...\n")
    test_loss, test_acc = model.evaluate(model_data['test']['inputs'], model_data['test']['labels'])

    # Save model if training was successful
    if test_acc > 0.85:
        model.save("models/bmi_model")
    else:
        print("\nModel failed to train!")


# Load model from file
def load_model():
    if os.path.exists('models/bmi_model'):
        bmi = tf.keras.models.load_model('models/bmi_model')

        return bmi
    else:
        print("\n\nERROR in load_model(): Model file not found")


# Classify user's BMI category as predicted by the model
def classify(pred):
    # BMI < 18.5
    if pred > 0.0 and pred <= 0.25:
        return 'underweight'
    
    # 18.5 < BMI < 24.9
    if pred > 0.25 and pred <= 0.5:
        return 'normal'

    # 25 < BMI < 29.9
    if pred > 0.5:
        return 'over'


# Build and train model
#build_model()

# Load model and predict bmi
model = load_model()
model._name = "DALE"

print(model._name)


weight = 199
height = 73
input = np.array([weight, height]).reshape(-1,2)
print("\nInput: " + str(input[0]) + "\n")
prediction = model.predict(input)

print(float(prediction[0]))