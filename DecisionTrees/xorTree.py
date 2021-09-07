"""
A Decision Tree created to 'learn' to act as an XOR gate
"""
from sklearn import tree
import numpy as np
import graphviz
import os

# Add graphviz to system path
os.environ["PATH"] += os.pathsep + "C:/Users/user/Documents/Classes/Fall 2021/Machine Learning/CodingPractice/DecisionTrees/venv/Lib/site-packages/graphviz/"

# Create training data and training labels
train_data = np.array([[0,0], [0,1], [1,0]])
train_labels = np.array([0, 1, 1])

# Create test data and labels
test_data = np.array([[1,1]])
test_label = np.array([1])

# Create decision tree
dt = tree.DecisionTreeClassifier()

# Train model on training data
dt = dt.fit(train_data, train_labels)



# Predict the class of test input
print("\nModel trained! Testing...\n")
print("Test data: " + str(test_data))
print("Predicted class: " + str(dt.predict(test_data)))
print("Actual class: " + str(test_label))
print("\nComputed probabilities for each class: " + str(dt.predict_proba(test_data)))

# # Visualize the trained model
# graph_data = tree.export_graphviz(dt, out_file=None)
# graph = graphviz.Source(graph_data)
# graph.render("XOR Classifier")