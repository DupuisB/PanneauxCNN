import random
import time

import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import cv2 as cv
from loaders.pano_loader import *
import utils.data_preprocessing as dp
from scipy import signal

path = "C:\\Users\\benyo\\PycharmProjects\\PanneauxCNN\\data\\EuDataset\\Testing\\001\\B_02294_00000.ppm"
path2 = "C:\\Users\\benyo\\PycharmProjects\\PanneauxCNN\\data\\EuDataset\\Classes"

mat1 = np.array([[9,4,1,2,2],[1,1,1,0,4],[1,2,1,0,6],[1,0,0,2,8],[9,6,7,4,6]])
mat2 = np.array([[0,2,1],[4,1,0],[1,0,1]])

#performs correlation product of mat1 and mat2 using scipy.signal.convolve2d
def convolve(mat1, mat2):
    return signal.correlate2d(mat1, mat2, mode='valid')
print(convolve(mat1, mat2))