"""
Basic project utilities
"""
import numpy as np

# Print a numpy array
def print_nparray(arr):
    
    print("(", end=" ")

    # Loop through arrays rows
    for row_idx in range(len(arr)):
        # Check row index for print formatting
        if row_idx == len(arr) - 1:
            str_elem = str(arr[row_idx])
            str_elem = str_elem.replace("\n", "")
            print(str_elem, end=" ")
        else:
            str_elem = str(arr[row_idx])
            str_elem = str_elem.replace("\n", "")
            print(str_elem + ", ", end=" ")

    print(")")
