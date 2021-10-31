"""
This file contains classes and functions for generating arbitrary or meaningful
datasets. Along with support for various visualization features.
"""
import numpy as np
import matplotlib.pyplot as plt

# Data generator
class DataGen:

    def __init__(self):
        self.fig = plt.figure()

        self.conversions = {
            'in2cm': 2.54,
            'lb2kg': 0.453592
        }

    # Generate a dataset to classify overweight patients based on height and weight
    def bmi_dataset(self, n_samples, show_plot=False):
        
        dataset = []
        dataset_plot = {
            'pos': [],
            'neg': []
        }

        max_weight = 350
        min_weight = 1
        max_height = 100
        min_height = 1

        ppl_weights = np.random.random_integers(low=min_weight, high=max_weight, size=n_samples)             # Weight in lbs
        ppl_heights = np.random.random_integers(low=min_height, high=max_height, size=n_samples)              # Height in inches

        metric_weights = ppl_weights * self.conversions['lb2kg']  # Weight in kg
        metric_heights = ppl_heights * self.conversions['in2cm']  # Height in cm


        # Loop through people and compute their BMI
        for i in range(n_samples):
            # Compute bmi
            bmi = (metric_weights[i] / metric_heights[i] / metric_heights[i]) * 10000.00

            # Check if patient is overweight
            if bmi >= 30.0:
                # Create array to store data pair (i.e. [weight,height,label])
                datapoint = [ppl_weights[i], ppl_heights[i], 1]
                dataset.append(datapoint)
                
                # Check for show plot
                if show_plot:
                    dataset_plot['pos'].append([ppl_weights[i], ppl_heights[i]])
            else:
                # Create array to store data pair (i.e. [weight,height,label])
                datapoint = [ppl_weights[i], ppl_heights[i], 0]
                dataset.append(datapoint)

                # Check for show plot
                if show_plot:
                    dataset_plot['neg'].append([ppl_weights[i], ppl_heights[i]])

        # Plot dataset
        if show_plot:

            bmi_plot = self.fig.add_subplot(111)

            np_pos = np.array(dataset_plot['pos'])
            np_neg = np.array(dataset_plot['neg'])

            # Plot positive points (Patients that are overweight)
            bmi_plot.scatter(np_pos[:,0], np_pos[:,1], c="red", marker="+", label="Overweight (BMI >= 30)")
            
            # Plot negative points (Patients that aren't overweight)
            bmi_plot.scatter(np_neg[:,0], np_neg[:,1], c="blue", marker="_", label="Not Overweight")

            p1 = [45,50]
            p2 = [250,77]

            x_vals = [p1[0],p2[0]]
            y_vals = [p1[1],p2[1]]

            bmi_plot.plot(x_vals, y_vals, c="green")

            bmi_plot.set_title("BMI Plot")
            bmi_plot.set_xlabel("Weight(lbs)")
            bmi_plot.set_ylabel("Height(in)")
            bmi_plot.legend(loc='best', bbox_to_anchor=(1.05,1.15))
            plt.show()

        return dataset