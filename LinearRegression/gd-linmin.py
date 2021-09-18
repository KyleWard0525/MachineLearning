"""
Implementation of the gradient descent algorithm to find the local minimum
of a linear function

The optimization problem is to find the value of x that results in the local
minimum of the function
"""

import numpy as np
import matplotlib.pyplot as plot
import time
import random

"""
Linear function

f(x) = 3x^2 / 4
"""
def f(x):
    return (3 * (x**2)) / 4


"""
Gradient of the linear function

grad(x) = f'(x) = 3x/2

NOTE: Just a single derivative as oppose to a 
vector of partial derivatives because the function only
has one parameter
"""
def gradient(x):
    return (3/2) * x


"""
Find the local minimum using gradient descent
"""
def gradient_descent(x, learning_rate, max_itrs):

    print("\nTRAINING PARAMETERS\n")
    print("Initial X (guess): " + str(x))
    print("Learning rate: " + str(learning_rate))
    print("Maximum training iterations: " + str(max_itrs) + "\n")
    
    curr_itr = 0                # Current training iteration
    step_size = 0          # Change between iterations

    if learning_rate >= 1:
        learning_rate = 1 / learning_rate

    # Train until solution is found or max iterations reached
    while f(x) > 0 and curr_itr < max_itrs:
        # Compute new value of x
        old_x = x
        x = x - learning_rate * gradient(x)
        step_size = (old_x - x)

        # Increment iteration
        curr_itr += 1

        # Print information every 10 iterations
        if curr_itr % 200 == 0:
            print("\nTraining iteration: " + str(curr_itr))
            print("Previous X: " + str(old_x))
            print("Updated X: " + str(x))
            print("Step size: " + str(step_size))
            print("f(x): " + str(f(x)))
            print("Minimum: " + str(0) + "\n")
            time.sleep(1)

    if f(x) == 0:
        print("Local Minimum at f(x) = 0 found in " + str(curr_itr) + " steps")
    elif curr_itr >= max_itrs:
        print("Maximum training iterations reached!\nX = " + str(x))



def main():
    init_x = random.randrange(5,100)
    learn_rate = 0.001
    max_itrs = 10000

    gradient_descent(init_x, learn_rate, max_itrs)

main()    
