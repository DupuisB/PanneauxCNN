import os

import numpy as np
import pickle
from PIL import Image, ImageOps, ImageEnhance
import utils.classes as cl
import random
from loaders.pano_loader import *

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
base_dir = os.path.join(data_dir, 'EuDataset')
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Testing')


def create_data(dir):
    """Transforme le dossier original EuDataset
    :return: numpy array de dimensions (nb_images, 32, 32, 3, 1)
    """
    sub_dirs = os.listdir(dir)
    data = []
    for sub_dir in sub_dirs:
        print(int(sub_dir))
        sub_dir_path = os.path.join(dir, sub_dir)

        # For a file to be an image, we have to make sure it ends with .ppm
        images = [os.path.join(sub_dir_path, image) for image in os.listdir(sub_dir_path) if image.endswith('.ppm')]
        for image in images:
            img = Image.open(image)
            img = img.resize((32, 32))
            img = np.expand_dims(np.array(img), axis=-1)
            img[0, 0, 0, 0] = int(sub_dir)
            data.append(img)
    return np.array(data)


def update_noir_et_blanc(data: np.array):
    """Transforme un jeu de donnee
    Remarque: Transforme un jeu de donnee de la forme (nb_images, 32, 32, 3, 1)
    :return: np.array transformee (noir et blanc, LHE)
    """

    # Creer un nouveau tableau de la forme adaptée
    new_data = np.empty((len(data), 32, 32, 1, 1))

    for i in range(len(data)):
        if i % 1000 == 0: print(i)
        img, label = data[i, :, :, :, 0], data[i, 0, 0, 0, 0]
        img = Image.fromarray(img)

        # Noir et blanc
        img = img.convert('L')

        # LHE
        img = ImageOps.equalize(img)

        # Normalisation (Calculs plus rapide)
        img = np.asarray(img)
        img = img / 255
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)

        img[0, 0, 0, 0] = label

        new_data[i] = img

    return new_data

def rdm_rotation(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec rotation aléatoire dans [-10, 10]
    """
    valeurs = [i for i in range(-10, 11)] # En degrés
    return image.rotate(np.random.choice(valeurs))

def rdm_brightness(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec luminosite augmentee (entre *0.5 et *1.5)
    """
    amount = np.random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(image).enhance(amount)

def rdm_flip(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image, symetrie verticale de l'entree
    """
    if random.random() < 0.5:  # Plus rapide que bool(random.getrandbits(1)) et que random.choice([True, False])
        return image
    return ImageOps.mirror(image)

def rdm_color(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec couleur augmentee (entre *0.8 et *1.2)
    """
    amount = np.random.uniform(0.7, 1.3)
    return ImageEnhance.Color(image).enhance(amount)

def rdm_contrast(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec contraste augmentee (entre *0.8 et *1.2)
    """
    amount = np.random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(image).enhance(amount)

def rdm_sharpness(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec nettete augmentee (entre *0.8 et *1.2)
    """
    amount = np.random.uniform(0.7, 1.3)
    return ImageEnhance.Sharpness(image).enhance(amount)

def rdm_zoom(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec zoom aleatoire (entre 0.8 et 1.2)
    """
    f = np.random.uniform(1, 1.2)
    # Calculate new dimensions
    width, height = image.size
    new_width = int(width * f)
    new_height = int(height * f)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Calculate crop box dimensions
    left = (resized_image.width - width) // 2
    top = (resized_image.height - height) // 2
    right = (resized_image.width + width) // 2
    bottom = (resized_image.height + height) // 2

    # Crop the image
    return resized_image.crop((left, top, right, bottom))


def salt(image: Image.Image, amount=0.05):
    """Salt and Pepper noise (Chaque pixel a une proba d'etre detruit et est remplacer par du blanc ou du noir (1/2))
    :param image: PIL.Image
    :return: PIL.Image avec bruit
    """
    image = np.asarray(image)
    modified = np.copy(image)
    shape = image.shape
    nb_pixel = np.prod(shape)
    for i in range(round(amount * nb_pixel)):
        x = np.random.randint(0, shape[0])
        y = np.random.randint(0, shape[1])
        modified[x, y] = np.random.choice([0, 255])

    return Image.fromarray(modified)

def update_classes(data: np.array, dico):
    """Met a jour les labels avec un dico = {ancien_num, nouveau_num}
    :return: Nouvelles donnees)
    """
    a_supprimer = 0
    for i in range(len(data)):
        label = int(data[i,0,0,0,0])
        if label not in dico.keys():
            a_supprimer += 1
    new_data = np.empty((len(data) - a_supprimer, 32, 32, 3, 1))
    decalage = 0

    for i in range(len(data)):
        label = int(data[i,0,0,0,0])
        if label in dico.keys():
            new_data[i - decalage] = data[i]
            new_data[i - decalage, 0, 0, 0, 0] = dico[label]
        else:
            decalage += 1
    print(f'{a_supprimer} images supprimées.')
    return new_data

def save(train, test, name):
    """Save as usual data
    Compile les donnees dans un dictionnaire et l'enregistre
    :param train: np.array
    :param test: np.array
    """
    dico = {}
    classes = cl.classes_EUD_fr()
    dico['labels'] = list(classes.values())
    dico['train'] = train
    dico['test'] = test
    with open(os.path.join(data_dir, name + '.pickle'), 'wb') as f:
        pickle.dump(dico, f)
    print('Data saved !')

def augmente_classe(data: np.array, label: int, nb: int):
    """Augmente le nombre d'images d'une classe, applique egalement LHE
    :param data: np.array Donnees, format classique
    :param label: int
    :param nb: int Nombre d'images voulues
    :return: np.array Donnees augmentees (Cette classe est augmentee)
    """
    images = [i for i in range(len(data)) if data[i, 0, 0, 0, 0] == label]

    if len(images) > nb:
        #Retourne exactement nb images
        return np.array([data[i] for i in np.random.choice(images, nb)])

    new_data = np.zeros((nb, 32, 32, 3, 1))

    for i in range(nb):
        x = np.random.choice(images)
        img = Image.fromarray(data[x, :, :, :, 0].astype(np.uint8))

        #Brightness
        img = rdm_brightness(img)
        #Color
        img = rdm_color(img)
        #Zoom
        img = rdm_zoom(img)
        #Contrast
        img = rdm_contrast(img)
        #Sharpness
        img = rdm_sharpness(img)

        #Remet le label
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)
        img[0, 0, 0, 0] = data[x, 0, 0, 0, 0]

        new_data[i] = img
    return new_data.astype(np.uint8)

def save_useless_now():
    with open("C:\\Users\\benyo\\Desktop\\train.pickle", 'rb') as f:
        train = pickle.load(f)
    with open("C:\\Users\\benyo\\Desktop\\test.pickle", 'rb') as f:
        test = pickle.load(f)
    save(train, test, 'EUD_RGB_ORIGINAL')

    ancien_to_nouveau = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 7: 5, 8: 6, 11: 7, 12: 8, 13: 9, 16: 10, 17: 11, 18: 12, 20: 13,
                         21: 14, 22: 15, 24: 16, 32: 17, 33: 18, 35: 19, 36: 20, 37: 21, 41: 22, 42: 23, 50: 24, 51: 25,
                         54: 26, 55: 27, 57: 28, 59: 29, 60: 30, 61: 31, 62: 32, 63: 33, 64: 34, 66: 35, 68: 36, 70: 37,
                         71: 38, 72: 39, 73: 40, 74: 41, 77: 42, 80: 43, 83: 44, 84: 45, 85: 46, 86: 47, 87: 48, 88: 49,
                         90: 50, 91: 51, 93: 52, 100: 53, 105: 54, 106: 55, 109: 56, 110: 57, 112: 58, 119: 59, 139: 60,
                         140: 61, 148: 62, 149: 63, 159: 64, 160: 65}

    data = EUD_loader_RGB()
    train = data[0]
    test = data[1]
    test = update_classes(test, ancien_to_nouveau)
    train = update_classes(train, ancien_to_nouveau)
    save(train, test, 'EUD_ORIGINAL_REDUIT')

    train, test = EUD_loader_ORIGINAL_REDUIT()
    new_train = []
    for i in range(65):
        print(i)
        increased = augmente_classe(train, i, 2000)
        new_train.append(increased)
    new_train = np.concatenate(new_train)
    save(new_train, test, 'EUD_RGB_AUGMENTE')

if __name__ == '__main__':
    train, test = EUD_loader_RGB()
    train = update_noir_et_blanc(train.astype(np.uint8))
    test = update_noir_et_blanc(test.astype(np.uint8))
    save(train, test, 'EUD_GREY_255_LHE')