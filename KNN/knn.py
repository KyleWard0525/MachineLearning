"""
K-Nearest Neighbor algorithm
"""
import math
import numpy as np

class KNN:

    # Initialize algorithm
    def __init__(self, dataset, k):
        self.data = dataset
        self.k = k

    # Squared Euclidean distance
    def euclid_dist(self, x, y):
        distance = 0.0 

        # Loop through input vectors (ignore last element as it should be class)
        for i in range(len(x) - 1):
            distance += (x[i] - y[i])**2

        return distance

    # Get K-Nearest Neighbors
    def neighbors(self, test_row):
        distances = []
        neighbors = []

        # Loop through training data
        for row in self.data:
            # Compute distances between test row and all rows in dataset
            d = self.euclid_dist(test_row, row)

            # Store distance from row
            distances.append([row,d])

        # Sort distances
        distances.sort(key = lambda x: x[1])

        # Find nearest neighbors
        for i in range(self.k):
            neighbors.append(distances[i][0])

        return neighbors

    # Make a prediction on test row based on known dataset
    def predict(self, test_row):
        # Get k-nearest neighbors
        neighbors = self.neighbors(test_row)

        # Get all classes of data from end of data vector
        outputs = [row[-1] for row in self.data]

        # Return class with the highest frequency
        return max(set(outputs), key=outputs.count)



        

    
    


dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

knn = KNN(dataset, k=3)

r = dataset[0]

for row in dataset:
    print("Distance from first row: " + str(knn.euclid_dist(r, row)))

print("\n\n")

print("%d nearest neighbors to %s: " % (knn.k, str(r)))
neighbors = knn.neighbors(r)

for n in neighbors:
    print("n = " + str(n))


prediction = knn.predict(r)
actual = dataset[0][-1]

print("Test data: " + str(r))
print("Prediction: " + str(prediction))
print("Actual: " + str(actual))