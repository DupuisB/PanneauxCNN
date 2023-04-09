import os

import numpy as np
import pickle
from PIL import Image, ImageOps
import classes as cl

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


def update_data(data: np.array):
    """Transforme un jeu de donnee
    Remarque: images en noir et blanc, equalization a faire avant
    :return: np.array transformee (noir et blanc, normalisé, ...)
    """
    print(data.shape)

    # Creer un nouveau tableau de la forme adaptée
    new_data = np.empty((len(data), 32, 32, 1, 1))

    for i in range(len(data)):
        if i % 1000 == 0: print(i)
        img, label = data[i, :, :, :, 0], data[i, 0, 0, 0, 0]
        img = Image.fromarray(img)

        # Noir et blanc
        img = img.convert('L')

        # Normalisation (Calcul plus rapide)
        img = np.asarray(img)
        img = img / 255
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)

        # LHE
        img = ImageOps.equalize(img)


        img[0, 0, 0, 0] = label

        new_data[i] = img

    return new_data

def rotate(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image avec rotation aléatoire dans [-35, -5] U [5, 35]
    """
    valeurs = [i for i in range(-36, -10)] + [i for i in range(5, 36)]
    return image.rotate(np.random.choice(valeurs))

def flip(image: Image.Image):
    """
    :param image: PIL.Image
    :return: PIL.Image, symetrie verticale de l'entree
    """
    return ImageOps.mirror(image)

def salt(image: Image.Image, amount=0.05):
    """Salt and Pepper noise (Chaque pixel a une proba d'etre detruit et est remplacer par du blanc ou du noir (1/2))
    :param image: PIL.Image
    :return: PIL.Image avec bruit
    """
    image = np.asarray(image)
    modified = np.copy(image)
    shape = image.shape
    print(shape)
    nb_pixel = np.prod(shape)
    for i in range(round(amount * nb_pixel)):
        x = np.random.randint(0, shape[0])
        y = np.random.randint(0, shape[1])
        modified[x, y] = np.random.choice([0, 255])

    return Image.fromarray(modified)

def update_classes(data: np.array, dico):
    """Met a jour les labels avec un dico = {ancien_num, nouveau_num}
    :return: Rien (Edite en place)
    Remarque: La complexité est infame, mais on ne fait ca qu'une fois
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

if __name__ == '__main__':
    #test = create_data(test_dir)
    #print(test.shape)
    #with open("C:\\Users\\benyo\\Desktop\\test.pickle", 'wb') as f:
    #    pickle.dump(test, f)

    #with open("C:\\Users\\benyo\\Desktop\\test.pickle", 'rb') as f:
    #    test = pickle.load(f)
    #print(test.shape)
    #test = update_data(test)

    #ancien_to_nouveau = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5, 10: 6, 11: 7, 12: 8, 14: 9, 15: 10, 16: 11, 17: 12,
    #                     19: 13, 20: 14, 21: 15, 22: 16, 23: 17, 24: 18, 31: 19, 32: 20, 34: 21, 35: 22, 36: 23,
    #                     37: 24, 40: 25, 50: 26, 51: 27, 53: 28, 56: 29, 58: 30, 60: 31, 63: 32, 64: 33, 65: 34,
    #                     66: 35, 67: 36, 68: 37, 69: 38, 70: 39, 71: 40, 73: 41, 79: 42, 80: 43, 81: 44, 82: 45,
    #                     83: 46, 84: 47, 85: 48, 86: 49, 87: 50, 89: 51, 90: 52, 92: 53, 99: 54, 104: 55, 105: 56,
    #                     108: 57, 118: 58, 142: 59, 145: 60, 146: 61, 148: 62, 158: 63, 160: 64}

    #with open("C:\\Users\\benyo\\Desktop\\test.pickle", 'rb') as f:
    #    test = pickle.load(f)
    #   new_test = update_classes(test, ancien_to_nouveau)
    #with open("C:\\Users\\benyo\\Desktop\\train.pickle", 'rb') as f:
    #    train = pickle.load(f)
    #    new_train = update_classes(train, ancien_to_nouveau)
    #save(new_train, new_test, 'EUD_RGB')
    pass
