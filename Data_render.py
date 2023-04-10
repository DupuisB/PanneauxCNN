import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
from time import sleep
import csv
from loaders.pano_loader import *
from PIL import Image


def classes(data):
    """
    :return: dictionnaire avec les classes et leur nombre d'occurence
    """
    classes = {}
    for couple in data:
        label = int(couple[0, 0, 0, 0])
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1
    return classes

def classes_original():
    """
    :return: dictionnaire avec les classes et leur nombre d'occurence
    """
    with open('data/train.csv', 'r') as f:
        reader = csv.reader(f)
        array = np.array(list(reader))[1:,:] #on enlève la première ligne (noms des colonnes)

    classes = {}
    for couple in array:
        label = int(couple[6])
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1
    return classes

def graph_labels(labels, save = False, show = True, path = './images/', name = 'default', titre = 'None'):
    """
    Affiche / sauvegarde graphique des classes
    """
    print('Generating graph...')
    plt.clf() #on efface le graphique précédent
    plt.style.use('./utils/metropolis.mplstyle')
    rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}, \usepackage[lf]{FiraSans}, \usepackage{sfmath}'
    plt.bar(labels.keys(), labels.values())
    plt.title(titre)
    plt.xlabel('Numero classe')
    plt.ylabel('Nombre d\'images')
    if save:
        plt.savefig(f'{path}{name}.png')
        print('Saved!')
    if show:
        plt.show()

#data = pano_loader_grey()

#Labels avant homogeneisation
#graph_labels(classes_original(), show = False, save = True, name = 'labels_originaux', titre = 'Images par classes avant homogénéisation')

#Labels apres homogeneisation
#graph_labels(classes(data[0]), show = False, save = True, name = 'labels_finaux', titre = 'Images par classes après homogénéisation')

#Labels avant homogeneisation EuDataset
#with open("C:\\Users\\benyo\\Desktop\\train.pickle", 'rb') as f:
#    data = pickle.load(f)
#graph_labels(classes(data), show = True, save = False, name = 'labels_originaux_EuDataset', titre = 'Images par classes avant homogénéisation')

train, test = EUD_loader_RGB()
#graph_labels(classes(train), show = True, save = False, name = 'labels_finaux_EuDataset', titre = 'Images par classes après homogénéisation')
Image.fromarray(train[121000, :, :, :, 0].astype(np.uint8)).show()
print(train[121000, 0, 0, 0, 0])
