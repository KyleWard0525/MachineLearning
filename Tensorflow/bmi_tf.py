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
from mlxtend.plotting import plot_decision_regions

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
model_data = preprocess(dataset, test_amt, normalize_inputs=False)

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
        tf.keras.layers.InputLayer(input_shape=(n_inputs,), name="input"),

        #   Hidden layer 1
        tf.keras.layers.Dense(units=n_hidden, kernel_regularizer=tf.keras.regularizers.L2(weight_reg), use_bias=True, name="hidden"),

        #   Output layer
        tf.keras.layers.Dense(n_outputs, activation=output_activation, name="output")
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


#   Load model from file
def load_model():
    if os.path.exists('models/bmi_model'):
        bmi = tf.keras.models.load_model('models/bmi_model')

        return bmi
    else:
        print("\n\nERROR in load_model(): Model file not found")


#   Classify user's BMI category as predicted by the model
def classify(pred):
    if pred == 0.0:
        return 'not overweight'
    else:
        return 'overweight'


#   Test model on a given weight and height
def test_bmi(model, height, weight):
    # Convert to metric units
    metric_weight = weight * datgen.conversions['lb2kg']
    metric_height = height * datgen.conversions['in2cm']

    # Compute user's bmi
    bmi = round((metric_weight / metric_height / metric_height) * 10000.00, 2)

    # Create input for model
    input = np.array([weight, height]).reshape(-1,2)

    # Make prediction
    prediction = float(model.predict(input)[0])

    # Classify user based on model's prediction
    classification = classify(prediction)

    print("\n\nWeight of " + str(weight) + " lbs.\nHeight of " + str(height) + " inches.\n" +
        "BMI of " + str(bmi) + "\nThe model has classified you as: '" + str(classification) + "'\n\n")


# Plot model results wrt all training samples
def plot_model(model):
    # Plot figure
    bmi_plot = plot.figure().add_subplot(111)

    # Plot points
    points = {
        'pos': [],
        'neg': [],

    }

    
    # Loop through datasets (train and test) in model_data
    for dataset in model_data:
        # Extract inputs and labels from data
        inputs = model_data[dataset]['inputs']
        labels = model_data[dataset]['labels']

        # Loop through each input-label pair
        for i in range(len(labels)):
            # Check if overweight or not
            if labels[i] == 1:
                # Add overweight sample to positive points
                points['pos'].append(inputs[i])
            else:
                # Add non-overweight sample to negative points
                points['neg'].append(inputs[i])

    print("\nPositive samples: " + str(len(points['pos'])))
    print("Negative samples: " + str(len(points['neg'])))

    np_pos = np.array(points['pos'])
    np_neg = np.array(points['neg'])

    # Plot positive points (Patients that are overweight)
    bmi_plot.scatter(np_pos[:,0], np_pos[:,1], c="red", marker="+", label="Overweight")
            
    # Plot negative points (Patients that aren't overweight)
    bmi_plot.scatter(np_neg[:,0], np_neg[:,1], c="blue", marker="_", label="Not Overweight")

    # Configure plot
    bmi_plot.set_title("BMI Plot")
    bmi_plot.set_xlabel("Weight(lbs)")
    bmi_plot.set_ylabel("Height(in)")
    bmi_plot.legend(loc='best', bbox_to_anchor=(1.05,1.15))

    # Display plot
    #plot.show()



# Load model and predict bmi
model = load_model()
model._name = "DALE"


print("\n\n" + str(model.summary()) + "\n\n")


print("\n\nPLOTTING MODEL...\n\n")

test_inputs = np.array(model_data['test']['inputs'])
test_labels = np.array(model_data['test']['labels'])

model.evaluate(test_inputs, test_labels)
plot_model(model)
plot_decision_regions(test_inputs, test_labels, clf=model, legend=2)
plot.show()



# Test case
# weight = 185
# height = 72
# test_bmi(model, height, weight)
# model_weights = model.get_layer(index=0).get_weights()
# model_biases = model.get_layer(index=1).get_weights()