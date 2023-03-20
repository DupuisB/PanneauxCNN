import pickle
import random
import time
import numpy as np
from PIL import Image
import os


# avg size: 44*42
# 60*60: 18860/99384 + grand
# data: 60497 / 21930

def pano_loader(x, size_train=None, size_test=None):
    """train, test en format [(entree), (sortie)] numpy array"""
    print('loading data')
    with open(f'./data/train_inputs_{x}', 'rb') as f:
        train_inputs = pickle.load(f)
    with open(f'./data/test_inputs_{x}', 'rb') as f:
        test_inputs = pickle.load(f)
    if size_train:
        random.shuffle(train_inputs)
        train_inputs = train_inputs[:size_train]
    if size_test:
        random.shuffle(test_inputs)
        test_inputs = test_inputs[:size_test]
    print('data loaded')
    return train_inputs, test_inputs


def pano_loader_64():
    return pano_loader(64)
def pano_loader_32():
    return pano_loader(32)

def pano_loader_small_64():
    return pano_loader(64, size_train=1000, size_test=300)
def pano_loader_small_32():
    return pano_loader(32, size_train=1000, size_test=300)

def pano_loader_medium_64():
    return pano_loader(64, size_train=10000, size_test=3000)
def pano_loader_medium_32():
    return pano_loader(32, size_train=10000, size_test=3000)


def vectorized_result(length, j):
    e = np.zeros((length, 1))
    e[j] = 1.0
    return e


def new_data(name):
    """
    training_data: [(entree, vecteur sortie)], test_data: [(entree, vecteur sortie)]
    """
    start = time.time()

    classes = 164
    path_train = './data/EuDataset/training/'
    path_test = './data/EuDataset/testing/'

    train_inputs, test_inputs = [], []

    for i in range(1, classes + 1):
        print(i)
        file_name = f"{i:0>3}"
        try:
            train_dir = os.listdir(os.path.join(path_train + file_name))
            test_dir = os.listdir(os.path.join(path_test + file_name))
        except:
            continue

        sortie = vectorized_result(classes, i - 1)

        for image in train_dir:
            # Choisir le format, convert('L') noir et blanc, resize normalise la forme
            train_inputs.append((np.asarray(Image.open(f"{path_train}{file_name}/{image}").resize((32, 32))), sortie))

        for image in test_dir:
            test_inputs.append((np.asarray(Image.open(f"{path_test}{file_name}/{image}").resize((32, 32))), sortie))

    print('Fin de la conversion, debut de l\'enregistrement...')

    # Enregistre le resultat obtenu
    with open(f'../data/train_inputs_{name}', 'wb') as f:
        pickle.dump(train_inputs, f)
    with open(f'../data/test_inputs_{name}', 'wb') as f:
        pickle.dump(test_inputs, f)

    print(f'Fini! Fichiers sauvés! \nDuration: {round(time.time() - start)} secondes.')
