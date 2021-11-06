"""
A CNN built with tensorflow to detect horses in images and generate a mask
image from the output matrix

kward60
"""

import os, sys, time, math,cv2
import numpy as np
from numpy.core.fromnumeric import ndim
from numpy.lib.arraypad import pad
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Imager import Imager


#   CNN Class
class HorseCNN:

    def __init__(self):
        # Initialize class variables
        self.img_path = "../images/horses/images/image-"                            #   Path to samples
        self.mask_path = "../images/horses/masks/mask-"                             #   Path to labels
        self.n_imgs = 328                                                           #   Total number of images/masks
        self.imgr = Imager(self.img_path + str(0) + ".png")                         #   Image processing helper

        # Load images and masks
        images, masks = self.loadDatasets()
        
        # Initialize model hyperparameters
        self.n_hidden = 128                                                         #   Number of hidden nodes
        self.n_outputs = self.imgr.n_pixels                                         #   Number of outputs is same size as input
        self.hidden_activation = 'relu'                                             #   ReLU activatoin function for hidden layers
        self.output_activation = 'sigmoid'                                          #   Sigmoid activation for output layer (0 = black pixel, 1 = white pixel)
        self.optim_algo = 'adam'                                                    #   Adam optimization algorithm
        self.lr = 0.0001                                                            #   Learning rate
        self.m1_decay = 0.9                                                         #   Exponential decay rate for 1st momentum estimates (aka beta_1)
        self.m2_decay = 0.999                                                       #   Exponential decay rate for 2st momentum estimates (aka beta_2)
        self.stabilizer = 1e-07                                                     #   Constant for numerical stability (aka epsilon)
        self.useAMS = True                                                          #   Whether or not to use AMSGrad variant of Adam (can help with avoiding large steps during training)
        self.n_filters = 64                                                         #   Number of convolutional filters to apply/train
        self.filt_shape = [3,3]                                                     #   Shape of convolutional filter(s)
        self.max_pool_shape = [2,2]                                                 #   Shape of matrix for Max Pooling Layer (For extracting sharp parts of the image?)
        self.avg_pool_shape = [4,4]                                                 #   Shape of matrix for Average Pooling Layer (For extracting smooth parts of the image?)


    #   Load image and mask datasets
    def loadDatasets(self):
        datasets = {
            'images': [],
            'masks': []
        }

        # Loop through and load images and masks
        for i in range(0, self.n_imgs):
            # Load image
            img, pixels = self.imgr.loadImage(self.img_path + str(i) + ".png")
            datasets['images'].append({'img': img, 'pixels': pixels})

            # Load mask
            img, pixels = self.imgr.loadImage(self.mask_path + str(i) + ".png")
            datasets['masks'].append({'img': img, 'pixels': pixels})


        return [datasets['images'], datasets['masks']]
        

    #   Convolve the current image
    def convolve(self, filter):
        return ndimage.convolve(self.imgr.pixels, filter)

    



horseAI = HorseCNN()
horseAI.loadDatasets()
