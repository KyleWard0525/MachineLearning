"""
A basic Decision Tree to learn how to process logic gates

Test labels:
- 0 = AND Gate
- 1 = OR Gate
"""
import numpy as np
import os
from utils import print_nparray
from scipy import stats

# train_data = np.array([[1,0],[0,1],[1,1],[0,0]])
# train_labels = np.array([0,0,1,1])
train_data = np.array([[0,0,0,0],[0,1,1,0],[1,1,0,0], # AND Data
[1,1,1,1],[1,0,1,1],[1,1,0,1]]) # OR Data

train_labels = np.array([0,0,0,1,1,1])

class DecisionTree:

    #   Initialize tree
    def __init__(self, t_data, t_labels):
        self.train_data = t_data
        self.train_labels = t_labels

        # For debugging
        self.gini_test_data = [[[1,0], [1,1]], [[1,1], [1,0]], # Should be 0.5
                              [[1,0], [1,0]], [[1,1], [1,1]]]  # Should be 0.0

        #   Get information about classes (labels)
        self.n_classes = self.numClasses()

        self.map_data()
        
    #   Compute the number of classes from training labels
    def numClasses(self):
        self.classes = []

        # Loop through training labels
        for label in self.train_labels:
            # Check if label is in unqiue_outputs
            if self.classes.count(label) == 0:
                self.classes.append(label)

        return len(self.classes)

    #   Map training data to training labels
    def map_data(self):

        self.data_map = {}

        # Loop through training data
        for i in range(len(self.train_data)):

            # Add key-value pair to data_map
            self.data_map.update({str(self.train_data[i]): self.train_labels[i]})

        print("Data Map: " + str(self.data_map))

    #   Split the given dataset into two groups based on a certain value
    def split(self, index, splitVal, dataset):
        left_partition = [] #   Left-hand side of the split
        right_partition = [] #  Right-hand side of the split

        # Check dataset type
        if not isinstance(dataset, np.ndarray):
            # Force dataset to be a numpy array
            dataset = np.array(dataset)

        # Check dataset shape
        if dataset.ndim > 2:
            num_rows = dataset.shape[0] * dataset.shape[1]
            num_cols = dataset.shape[2]
            dataset = dataset.reshape([num_rows, num_cols])

            print("Dataset hase been reshaped. New shape = " + str(dataset.shape))

        #   Loop through all rows in the dataset
        for row in dataset:

            if splitVal == 0:
                if row[index] <= splitVal:
                    left_partition.append(row)          #   Add row to left-hand side
                else:
                    right_partition.append(row)         #   Add row to right-hand side
            else:
                if row[index] < splitVal:
                    left_partition.append(row)          #   Add row to left-hand side
                else:
                    right_partition.append(row)         #   Add row to right-hand side


        return [left_partition, right_partition]

    #   Find the best split for the given dataset
    def best_split(self, dataset):
        best_idx, best_val, best_score, best_subset = 1000,1000,1000,None

        # Loop through indices of a row in dataset
        for idx in range(len(dataset[0])):
            # Loop through all rows of the dataset
            for row in dataset:
                
                # Split the dataset in all possible ways and evaluate the best split
                split_data = self.split(idx, row[idx], dataset)
                gini = self.gini(split_data, self.classes)

                # print("\nSplit Info:\nindex = " + str(idx) + "\nvalue = " + str(row[idx]) + "\nGini = " + str(gini) +
                # "\nsplit = " + str(split_data) + "\n")

                # Check gini score against best score
                if gini < best_score:
                    # Set new node properties
                    best_idx = idx
                    best_val = row[idx]
                    best_score = gini
                    best_subset = split_data


        # Return information about best split (new node)
        return {
            "index": best_idx,
            "value": best_val,
            "gini": best_score,
            "split": best_subset 
        }

    #   Compute a split's Gini index
    def gini(self, split_data, classes):
        gini = 0.0

        # Compute the total number of elements in each subset of split data
        n_elems = 0

        for rows in split_data:
            n_elems += float(len(rows))


        # Loop through split_data
        for subset in split_data:
            size = float(len(subset))

            # Protect against zer0-division
            if size == 0:
                continue
            
            row_score = 0.0

            # Loop through classes
            for class_type in classes:

                # Score the subset based on score for each class
                proportion = [row[-1] for row in subset].count(class_type) / size

                # Compute score as squared error of proportion
                row_score += proportion*proportion
            
            # Weight the score of the row relative to its size
            gini += (1.0 - row_score) * (size / n_elems)

        return gini


    # Create a leaf node by extracting the attribute that occurrs most
    def leaf_node(self, data):
        counts = [0,0,0]

        for row in data:
            # increment counts
            counts[self.data_map[str(row)]] += 1

        leaf_value = np.argmax(np.array(counts))

        print("Leaf value: " + str(leaf_value))

        return leaf_value



    # Create the child node's of the tree
    def create_nodes(self, curr_node, max_depth, min_size, curr_depth):

        # Extract left and right partitions from current node
        left_part, right_part = curr_node["split"]

        print("Left partition: " + str(left_part) + "\nRight partition: " + str(right_part))

        # Remove partitions from current node
        del(curr_node["split"])

        # Check if either partition is empty
        if not left_part or not right_part:
            # Create single leaf node as there is no splitting left to do and only one branch exists

            curr_node["left"] = curr_node["right"] = self.leaf_node(left_part + right_part)
            return

        # Check if max tree depth has been reached
        if curr_depth >= max_depth:
            # Create left and right leaf nodes and return
            curr_node["left"] = self.leaf_node(left_part)
            curr_node["right"] = self.leaf_node(right_part)
            return

        # Build left side of the tree
        # Check size of left partition
        if len(left_part) <= min_size:
            # Left node is too small to be split again. Make it a leaf node
            curr_node["left"] = self.leaf_node(left_part)
        else:
            # Get the best split for the partition
            curr_node["left"] = self.best_split(left_part)

            # Recursively build left side of the tree
            self.create_nodes(curr_node["left"], max_depth, min_size, curr_depth + 1)

        # Build right side of the tree
        if len(right_part) <= min_size:
            # Right node to small to split. Make leaf node
            curr_node["right"] = self.best_split(right_part)

            # Recursively build right side of the tree
            self.create_nodes(curr_node["right"], max_depth, min_size, curr_depth + 1)

    # Build the decision tree from the given training dataset
    def build_tree(self, max_depth, min_size):
        self.root_node = self.best_split(self.train_data) # Create root node with initial split

        # Recursively build the tree into the root node
        self.create_nodes(self.root_node, max_depth, min_size, 1)

        # Return tree
        return self.root_node


    # Make a prediction using the decision tree
    def predict(self, data_row, curr_node):
            # Left side
            if data_row[curr_node["index"]] < curr_node["value"]:
                # Check if current node is traversable or leaf

                # Traversable
                if isinstance(curr_node["left"], dict):
                    # Recursively run data through left side of the tree
                    return self.predict(data_row, curr_node["left"])
                # Leaf node
                else:
                    return curr_node["left"]

            # Right side
            else:
                if "right" in curr_node:
                    # Traversable
                    if isinstance(curr_node["right"], dict):
                        # Recursively run data through left side of the tree
                        return self.predict(data_row, curr_node["right"])
                    # Leaf node
                    else:
                        return curr_node["right"]
                else:
                    return self.predict(data_row, curr_node["left"])

    

# Run test set on tree
def test_tree(tree_obj, tree_model, test_data, test_labels):
    print("\n\nRunning test data on decision tree...\n")

    # Check test data
    if not isinstance(test_data, np.ndarray):
        print("Forcing test_data to numpy array")
        # Force test data to numpy array
        test_data = np.array(test_data)

    # Test variables
    total_samples = test_data.shape[0]
    correct_predictions = 0
    accuracy = 0.0

    print("Test data: " + str(test_data))

    # Iterate through test data
    for idx in range(len(test_data) - 1):
        
        prediction = tree_obj.predict(test_data[idx], curr_node=tree_model)
        actual = test_labels[idx]

        # Check prediciton
        if prediction == actual:
            # Update test vars
            correct_predictions += 1
            accuracy = abs(correct_predictions / total_samples)
        else:
            correct_predictions = 0
            accuracy = 0.0


        print("\nTest #%d: data = %s\nPredicted class: %s\nActual class: %s" % (idx, str(test_data[idx]), str(prediction), str(actual)))

    
    # Return test results
    return {'correct': correct_predictions, 'accuracy': accuracy}



# Main function
def main():
    # Create decision tree and pass in training data
    dt = DecisionTree(train_data, train_labels)
    print("Decision tree classes: " + str(dt.classes))

    max_depth = 2               # Maximum tree depth
    min_size = 3                # Minimum size of a row

    tree_model = dt.build_tree(max_depth, min_size)

    print("Tree model: " + str(tree_model))

    # Make a prediciton with the tree using test data

    test_data = np.array([[0,0,1,1],[1,0,0,0],[0,1,1,1],[0,0,0,1],[1,0,0,1],[1,1,0,0]])
    test_labels = np.array([1,0,1,0,1,0])

    results = test_tree(dt, tree_model, test_data, test_labels)

    print("\n\nTest Results:\n")
    print("Total number of samples: " + str(test_data.shape[0]))
    print("Number of correct predictions: " + str(results["correct"]))
    print("Accuracy: %.2f" % results["accuracy"])

    
main()




