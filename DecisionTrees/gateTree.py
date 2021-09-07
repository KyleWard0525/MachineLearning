"""
A decision tree for using logic gates
"""
import numpy as np
from sklearn import tree
import graphviz
import os

# Add graphviz to system path
os.environ["PATH"] += os.pathsep + "C:/Users/user/Documents/Classes/Fall 2021/Machine Learning/CodingPractice/DecisionTrees/venv/Lib/site-packages/graphviz/"

train_data=[[1,0,0,0],[1,1,1,1],[1,0,1,0],[1,1,0,0], # AND Data
[1,0,0,1],[1,1,1,1],[1,0,1,1],[1,1,0,1], # OR Data
[1,0,0,1],[1,1,1,0],[1,0,1,0],[1,1,0,0]] # XOR Data

train_labels=[0,0,0,0,1,1,1,1,2,2,2,2]

test_data=[[0,0,0,1],[0,1,1,0],[0,1,0,0], # AND Data
[0,0,0,0],[0,1,1,1],[0,1,0,1], # OR Data
[0,0,0,0],[0,1,1,0],[0,1,0,1]] # XOR data

test_labels=[0,0,0,1,1,1,2,2,2]

# Create decision tree
d_tree = tree.DecisionTreeClassifier()

# Train model on data
d_tree.fit(train_data, train_labels)

# Predict the class of test input
print("\nModel trained! Testing...\n")
print("Test data: " + str(test_data))
print("Predicted class: " + str(d_tree.predict(test_data)))
print("Actual class: " + str(test_labels))
print("\nComputed probabilities for each class: " + str(d_tree.predict_proba(test_data)))