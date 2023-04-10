import random
import time

import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import cv2 as cv
from loaders.pano_loader import *
import utils.data_preprocessing as dp

path = "C:\\Users\\benyo\\PycharmProjects\\PanneauxCNN\\data\\EuDataset\\Testing\\001\\B_02294_00000.ppm"
path2 = "C:\\Users\\benyo\\PycharmProjects\\PanneauxCNN\\data\\EuDataset\\Classes"

train, test = EUD_loader_ORIGINAL_REDUIT()
train2, test2 = pano_loader_grey()

print(train[0,0,0,0,0])
print(train2[0,0,0,0,0])