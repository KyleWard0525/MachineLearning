"""
This class is an implementation of the ID3 algorithm to build a decision tree
from a given set of labeled data

kward
"""

import re
import math
import numpy as np

# ID3 Algorithm for Decision Trees
class ID3:

    # Initialize algorithm
    def __init__(self, data_file=None):
        
        # Check if a data filename was passed in
        if data_file:
            # Create class objects
            self.parser = Parser(data_file)
            self.probability = ID3_Probability(self.parser.data)

            # Set class fields
            self.data = self.parser.data
            self.data_labels = self.parser.data_labels
            self.attrNames = self.parser.vars
            self.attributes = [*range(0,len(self.data[0])-1)]
            self.trained = False

            # Compute initial metrics
            self.start_pdf = self.probability.pdf(self.data)
            self.start_entropy = self.probability.entropy(self.start_pdf)

        else:
            raise RuntimeError("\n\nError in ID3.__init__(): \n\n'WHERE'S THE DATA?!?!?!?!?!????????'\n\n\t-ID3 <3\n\n")


    # Split data based on a given attribute
    def split(self, attr, data):

        left = []                   # Left partition of the split
        right = []                  # Right partition of the split

        # Loop through rows in dataset
        for row in data:

            # Compare value of attribute in row with split value
            if row[attr] < 1:
                left.append(row)    # Add row to left partition
            else:
                right.append(row)   # Add row to right partition

        split = [left, right]

        # If a split results in an empty branch, set empty branch to none empty
        if not left:
            left_pdf = 0.0
            left_e = 0.0

            right_pdf = ID3_Probability.pdf(right)

            # Check for and remove zeros
            if 0.0 in right_pdf:
                right_pdf.remove(0.0)

            right_e = ID3_Probability.entropy(right_pdf)

            # Compute the entropy of the entire split
            weighted_avg_entropy = ((len(left) / len(data)) * left_e) + ((len(right) / len(data)) * right_e)

            split = {
            'left': left,
            'right': right,
            'left_entropy': left_e,
            'right_entropy': right_e,
            'wAvg_entropy': weighted_avg_entropy
            }

            return split

        elif not right:
            right_pdf = 0.0
            right_e = 0.0

            left_pdf = ID3_Probability.pdf(left)

            # Check for and remove zeros
            if 0.0 in left_pdf:
                left_pdf.remove(0.0)

            left_e = ID3_Probability.entropy(left_pdf)

            # Compute the entropy of the entire split
            weighted_avg_entropy = ((len(left) / len(data)) * left_e) + ((len(right) / len(data)) * right_e)

            split = {
            'left': left,
            'right': right,
            'left_entropy': left_e,
            'right_entropy': right_e,
            'wAvg_entropy': weighted_avg_entropy
            }

            return split

        

        # Compute probability distribution of classes on left and right branch
        left_pdf = ID3_Probability.pdf(split[0])
        right_pdf= ID3_Probability.pdf(split[1])

        # Check for and remove zeros
        if 0.0 in left_pdf:
            left_pdf.remove(0.0)
        if 0.0 in right_pdf:
            right_pdf.remove(0.0)

        # Compute the entropy of the left and right branch
        left_e = ID3_Probability.entropy(left_pdf)
        right_e = ID3_Probability.entropy(right_pdf)

        # Compute the entropy of the entire split
        weighted_avg_entropy = ((len(split[0]) / len(data)) * left_e) + ((len(split[1]) / len(data)) * right_e)

        split = {
            'left': left,
            'right': right,
            'left_entropy': left_e,
            'right_entropy': right_e,
            'wAvg_entropy': weighted_avg_entropy
        }

        return split

    # Find the best split for the given set of data
    def best_split(self, data):
    
        gains = []      # Information gain scores for each attribute split

        # Loop through all attributes to find attribute with best IG for best split
        for attr_val in self.attributes:
            attr_name = self.attrNames[attr_val]
            split = self.split(attr_val, data)
            wae = split['wAvg_entropy']
            info_gain = self.start_entropy - wae
            gains.append(info_gain)

        
        best = {
            'attr': gains.index(max(gains)),
            'split': self.split(gains.index(max(gains)),data)
        }

        return best

    # Create a leaf node
    def leaf_node(self, data):
        label_counts = [0,0]

        for row in data:
            if row[-1] == 0:
                label_counts[0] += 1
            elif row[-1] == 1:
                label_counts[1] += 1

        value = label_counts.index(max(label_counts))

        return value

    # Recursively build child nodes of the tree from the root node
    def create_nodes(self, parent_node, max_depth, min_part_size, curr_depth):

        # Create left and right partitions (branches) off of parent node
        left = parent_node['split']['left']
        right = parent_node['split']['right']

        # Remove paritions from parent node
        del(parent_node['split'])

        # Check if either partition is empty, if so, create a leaf node (BASE CASE)
        if not left or not right:
            parent_node['left'] = parent_node['right'] = self.leaf_node(left + right)
            return

        # Max tree depth reached. Create leaf nodes (BASE CASE)
        if curr_depth >= max_depth:
            parent_node['left'] = self.leaf_node(left)
            parent_node['right'] = self.leaf_node(right)
            return


        ### First, build the left side of the tree recursively ###

        # Check if left partition can be split again
        if len(left) <= min_part_size:
            # Create a leaf node from partition
            parent_node['left'] = self.leaf_node(left)

        # Partition can be split again
        else:
            # Get best split
            parent_node['left'] = self.best_split(left)

            # Recursively build the rest of the left side of the tree
            self.create_nodes(parent_node['left'], max_depth, min_part_size, curr_depth + 1)

        
        ### Next, build the right side of the tree ###
        
        # Partition can't be split again
        if len(right) <= min_part_size:
            # Create leaf node
            parent_node['right'] = self.leaf_node(right)
        
        # Partition can be split again
        else:
            # Get best split
            parent_node['right'] = self.best_split(right)

            # Recursively build the rest of the right side of the tree
            self.create_nodes(parent_node['right'], max_depth, min_part_size, curr_depth + 1)


    
    # Build decision tree from training data
    def build_tree(self, data, min_part_size, max_depth):
        
        # Create root node by getting the best initial split
        self.root_node = self.best_split(data)

        # Build the rest of the tree from the root node
        self.create_nodes(self.root_node, max_depth, min_part_size, 1)
        self.trained = True

        # Return tree
        return self.root_node


    # Make a prediction by running data through the tree
    @staticmethod
    def predict(data, curr_node):
        
        # Check which side of the tree to travel down based on root node's attribute
        if data[curr_node['attr']] < 1:

            # Traverse left side of the tree #

            # Check if left branch leads to a decision node or leaf node
            # Decision node
            if isinstance(curr_node['left'], dict):
                # Recursively trace down the left side of the tree
                return ID3.predict(data, curr_node['left'])
            
            # Leaf node
            else:
                return curr_node['left']
        else:

            # Traverse the right side of the tree #

            # Check if right branch leads to a decision node or leaf node
            # Decision node
            if isinstance(curr_node['right'], dict):
                # Recursively trace down the left side of the tree
                return ID3.predict(data, curr_node['right'])
            
            # Leaf node
            else:
                return curr_node['right']

    


# For parsing and cleaning data
class Parser:

    # Initialize parser
    def __init__(self, filename):
        self.data = self.read_data(filename)    # Full dataset
        self.vars = self.data[1]                # Attribute names (variables)
        self.data = self.data[0]                # Train/Test data
        self.get_labels()

    # Load data from a file
    @staticmethod
    def read_data(filename):
        f = open(filename, 'r')
        p = re.compile(',')
        data = []
        header = f.readline().strip()
        varnames = p.split(header)
        namehash = {}
        for l in f:
            data.append([int(x) for x in p.split(l.strip())])
        return (data, varnames)

    # Get data labels
    def get_labels(self):
        self.data_labels = []

        # Loop through all rows in data
        for row in self.data:
            # Add class label to parser's list before removal
            self.data_labels.append(row[-1])

    # Split dataset into data and labels
    def split_data_labels(self, dataset):
        labels = []
        data = []

        # Loop through rows in dataset
        for row in dataset:
            # Save label and remove from data row
            labels.append(row[-1])
            del(row[-1])

            # Save new data row
            data.append(row)

        return [data, labels]
        


# Probability Helper Class
class ID3_Probability:

    # Initialize object with a dataset
    def __init__(self, dataset):
        self.data = dataset

    # Find number of times an attribute is equal to a value
    def count_occurrs(self, attr, val):
        occurrs = 0

        # Loop through dataset and count occurences
        for row in self.data:
            if row[attr] == val:
                occurrs += 1

        return occurrs

    # Compute the probability distribution of a binary data vector
    # Assuming class label is in the last element
    @staticmethod
    def pdf(dataset):
        labels = [0,1]
        occurrs = [0,0]
        prob_dist = [0.0,0.0]

        # Loop through dataset
        for row in dataset:
            if row[-1] == 0:
                occurrs[0] += 1
            elif row[-1] == 1:
                occurrs[1] += 1

        # Compute probability distribution 
        prob_dist = [occurrs[0] / len(dataset), occurrs[1] / len(dataset)]
        
        return prob_dist

    

    # Compute entropy of a probability distribution
    @staticmethod
    def entropy(prob_dist):
        sum = 0.0

        # Loop through probability distribution
        for i in range(len(prob_dist)):
            sum += prob_dist[i] * math.log2(prob_dist[i])

        return -1.0 * sum


# Compute the accuracy of a tree
def accuracy(tree, data):

    n_samples = len(data)
    n_correct = 0

    for row in data:

        if ID3.predict(row, tree) == row[-1]:
            n_correct += 1

    return n_correct / n_samples

# Build tree for each training set using ID3 and test
def runAllTests():
    numSets = 3

    # Data files
    train_files = ["data_sets1/training_set.csv", "data_sets2/training_set.csv", "agaricuslepiotatrain1.csv"]
    validation_files = ["data_sets1/validation_set.csv", "data_sets2/validation_set.csv"]
    test_files = ["data_sets1/test_set.csv", "data_sets2/test_set.csv", "agaricuslepiotatest1.csv"]

    # Data sets
    train_sets = [Parser.read_data(train_files[0])[0], Parser.read_data(train_files[1])[0], Parser.read_data(train_files[2])[0]]
    test_sets = [Parser.read_data(test_files[0])[0], Parser.read_data(test_files[1])[0], Parser.read_data(test_files[2])[0]]
    valid_sets = [Parser.read_data(validation_files[0])[0], Parser.read_data(validation_files[1])[0]]

    # Decision trees
    trees = []

    # Training parameters
    min_partition_size = 1
    max_depth = 15

    for i in range(numSets):
        # Create instance of ID3 algorithm with training data
        id3 = ID3(train_files[i])

        # Build tree and append to list of trees
        tree = id3.build_tree(id3.data, min_partition_size, max_depth)

        # Compute tree's accuracy on training, validation, and test sets
        train_acc = accuracy(tree, train_sets[i])
        test_acc = accuracy(tree, test_sets[i])

        if i < 2:
            valid_acc = accuracy(tree, valid_sets[i])
        else:
            valid_acc = 'No validation file.'

        # Create struct of tree and info
        tree_info = {
            'id': i,
            'tree': tree,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'test_acc': test_acc
        }
        
        # Save to list of trees
        trees.append(tree_info)

        print("\n" + str(tree_info) + "\n")


def main():


    runAllTests()



    # tree = ID3(data_file="data_sets1/training_set.csv")
    # prob = ID3_Probability(tree.data)

    # start_pdf = prob.pdf(tree.data)
    # start_e = prob.entropy(start_pdf)
    # print("\n\nDataset initial pdf: " + str(start_pdf))
    # print("Dataset initial entropy: " + str(start_e) + "\n")

    # # Training parameters
    # train_data = tree.data
    # min_partition_size = 1
    # max_depth = 12

    # # Build the tree
    # root = tree.build_tree(train_data, min_partition_size, max_depth)

    # # Read in validation and test data
    # test_set = Parser.read_data('data_sets1/test_set.csv')[0]
    # valid_set = Parser.read_data('data_sets1/validation_set.csv')[0]

    # # Accuracy metrics
    # train_acc = accuracy(root, train_data)
    # valid_acc = accuracy(root, valid_set)
    # test_acc = accuracy(root, test_set)

    # print("\nTraining accuracy: %.3f" % train_acc)
    # print("Validation accuracy: %.3f" % valid_acc)
    # print("Test accuracy: %.3f" % test_acc)

    print("\n\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        quit()