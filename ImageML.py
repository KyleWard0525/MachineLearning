"""
This file contains classes and functions for reading in images and parsing
them to matrices to be used in machine learning tasks

kward60
"""
from functools import cache, cached_property
from PIL import Image
from sys import maxsize
import numpy as np
import sys
import os

# Image class for Machine Learning applications
class ImageML:

    # Initialize object
    def __init__(self, img_path):

        # Initialize class variables
        self.img_path = img_path                                        #   Image path
        self.img = Image.open(img_path)                                 #   Image object
        self.img_width, self.img_height = self.img.size                 #   Image shape
        self.n_pixels = self.img_width * self.img_height                #   Total number of pixels
        self.pixels = list(self.img.getdata())                          #   RGB values of each pixel

        # Image color groups
        self.colors = np.array(self.img.getcolors(maxcolors=int(self.n_pixels)), dtype='object')

        # Sort color groups by highest frequency
        self.colors = np.sort(self.colors, axis=0)

        # Create data directory if it doesn't exist
        if not os.path.exists('data/'):
            os.mkdir('data/')


    # Save ALL pixels to file
    def savePixels(self, filename="pixels.csv"):
        file = open('data/' + str(filename), "w")

        # Pixel counter
        px_cntr = 0

        # Loop through pixels
        for pixel in self.pixels:
            
            count = 0

            # Store pixel index
            file.write(str(px_cntr) + ", ")

            # Loop through pixel's rgb values
            for value in pixel:
                if count < len(pixel) - 1:
                    file.write(str(value) + ", ")
                else:
                    file.write(str(value))

                count += 1
            
            # Increment pixel counter
            px_cntr += 1
            file.write("\n")

        # Close file
        file.close()

        # Check if file was saved successfully
        if os.path.exists('data/' + filename):
            print("\nPixel data file saved successfully to: 'data/" + str(filename) + "'\n")
        else:
            print("\n\nERROR in savePixels(): failed to save file!")

    
    # Save image color groups to file in order of highest freq. to lowest freq.
    def saveColorGroups(self, filename="colors.csv"):

        file = open('data/' + str(filename), "w")

        # Loop through sorted colors
        for i in range(1, len(self.colors)):
            # Save color to file
            file.write(str(i-1) + ", " + str(self.colors[-i]))
            file.write("\n")

        file.close()

        # Check if file was saved successfully
        if os.path.exists('data/' + filename):
            print("\nColor data file saved successfully to: 'data/" + str(filename) + "'\n")
        else:
            print("\n\nERROR in saveColorGroups(): failed to save file!")


    # Print info about image
    def printImgInfo(self):
        print("\nImage path: " + str(self.img_path))
        print("Image Width: " + str(self.img_width) + " pixels")
        print("Image Height: " + str(self.img_height) + " pixels")
        print("Number of pixels: " + str(self.n_pixels))
        print("Number of unique colors: " + str(len(self.colors)))
        print("\n")


    # Get image pixels
    def getPixels(self):
        return self.pixels

    # Compute frequency of a color wrt all pixels
    def getColorFreq(self, color_idx):
        color = self.colors[color_idx]
        print(color)
        n_occur = color[0]
        
        freq = float(n_occur / len(self.colors))
        return freq

im = ImageMatrix("images/mona-lisa.jpg")
im.printImgInfo()
c_freq = im.getColorFreq(-1)
print("Frequency: " + str(c_freq))