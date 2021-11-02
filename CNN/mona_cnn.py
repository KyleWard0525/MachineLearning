"""
This file is an attempt at learning how a CNN works and how to build and
use one to recognize patterns in images

kward60
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Imager import Imager

# Crop just the head of the image
def crop_head(imgr):
    image_px = imgr.pixels
    head_arr = imgr.imgSubsection(imgr.img, 175, 50, 485, 500)
    return head_arr

# Image processing class
imgr = Imager("../images/minhattan.png")
imgr.printImgInfo()

# Working on just the cropped part (for now)
#cropped_matrix = crop_head(imgr)

# Create filter for convolution
filter_shape = [3,3,3]
conv_filter = np.random.randint(low=-1, high=1, size=filter_shape)
print("Convolutional filter = :" + str(conv_filter) + "\n")

conv_matrix = ndimage.convolve(imgr.pixels, conv_filter)

conv_img = Image.fromarray(conv_matrix)
conv_img.save("data/minhattan_meg.png")