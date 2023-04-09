import random
import time

import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import cv2 as cv

path = "C:\\Users\\benyo\\PycharmProjects\\PanneauxCNN\\data\\EuDataset\\Testing\\001\\B_02294_00000.ppm"
path2 = "C:\\Users\\benyo\\PycharmProjects\\PanneauxCNN\\data\\EuDataset\\Classes"

array = np.zeros((16,14,13,12))
print(array.shape)
array = np.delete(array, 0, 0)
print(array.shape)