from network import *
# noinspection PyUnresolvedReferences
from utils.activation_functions import *
from loaders.pano_loader import *
from layers import *

train, test = pano_loader_grey()
net = Network(loader=pano_loader_grey, layers=[Convolution((32, 32, 1), 4, 16, sigmoid),
                                               Reshape((29, 29, 16), (29 * 29 * 16, 1)),
                                               Dense(29 * 29 * 16, 43, sigmoid), Softmax()], cost=CrossEntropyCost)
