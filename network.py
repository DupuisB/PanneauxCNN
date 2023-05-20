import numpy as np
import pickle

from layers import *
from utils.classes import *
from utils.activation_functions import *
from utils.cost_functions import *
from loaders.pano_loader import *

class Network(object):

    def __init__(self, loader, layers: list, nb_classes=43, cost_function=DistanceEuclidienne2, classes=classes_fr()):
        """
        :param layers: [Layer1, Layer2, ...]
        """
        self.classes = classes
        self.layers = layers
        self.loader = loader
        self.nb_classes = nb_classes
        self.num_layers = len(layers)
        self.cost = cost_function
        self.monitor = ([],[])

    def feedforward(self, entree):
        """
        Renvoie le vecteur sortie pour un vecteur d'entree entree
        :return: np.array a 1 dimension
        """
        sortie = entree
        for layer in self.layers:
            sortie = layer.feedforward(sortie)
        return sortie

    def feedforwardClasse(self, entree):
        """
        Renvoie la classe pour un vecteur d'entree a
        :return: string
        """
        sortie = self.feedforward(entree)
        return self.classes[np.argmax(sortie)]

    def train(self, epochs=1, eta=2,
              test_accuracy=False, train_accuracy=False,
              mini_batch_size=20):
        """
        Entraine et retourne les evaluations si demandées
        :return: (train_accuracy, test_accuracy) (listes d'entiers a une dimension)
        """

        train, test = self.loader()
        start = time.time()

        if train_accuracy: self.monitor[0].append(self.accuracy(train, '(train)')[0])
        if test_accuracy: self.monitor[1].append(self.accuracy(test, '(test)')[0])

        for epoch in range(epochs):

            # train, test = loader()
            n_train = len(train)
            print(f'Epoch {epoch + 1} started...')

            np.random.shuffle(train)

            batches = [
                train[k:k + mini_batch_size]
                for k in range(0, n_train, mini_batch_size)]

            monitor_batch = 0  #
            for batch in batches:
                if monitor_batch % 1000 == 0:  #
                    print(f'Batch: {monitor_batch}')  #
                self.train_batch(batch, eta)
                monitor_batch += 1  #
            print(f"\nEntrainement epoch {epoch + 1} fini")
            print(f"Only required {time.time() - start:.2f}s")

            if train_accuracy: self.monitor[0].append(self.accuracy(train, '(train)')[0])
            if test_accuracy: self.monitor[1].append(self.accuracy(test, '(test)')[0])

        return train_accuracy, test_accuracy

    def train_batch(self, batch, eta):
        batch_size = len(batch[0])
        for entree in batch:
            image, label = entree[:, :, :, 0], delta_vecteur(int(entree[0, 0, 0, 0]), self.nb_classes)
            output = self.feedforward(image)
            nabla = self.cost.prime(output, label)
            for layer in reversed(self.layers):
                nabla = layer.backpropagation(eta, nabla)

        for layer in reversed(self.layers):
            layer.backpropagation_update(eta, batch_size)
            layer.reset_backpropagation()

    def accuracy(self, data, add_str=''):
        """
        :param data: np.array de taille (nb_images, image, etiquette)
        :return: (nb_juste:int, nb_total:int)
        """
        print('Evaluating accuracy...')
        total = len(data)
        juste = 0
        for couple in data:
            juste += int(np.argmax(self.feedforward(couple[:, :, :, 0])) == int(couple[0, 0, 0, 0]))
        print(f"Précision: {juste}/{total} ou {juste / total:.4%} {add_str}")
        return juste, total

    def save(self, nom):
        """"
        Sauvegarde le reseaux sous ../network/{nom}
        """
        with open('./networks/' + nom, "wb") as f:
            pickle.dump(self, f)
            f.close()
        print(f'Network {nom} saved')


def load(nom):
    """
    Charge le reseau {nom}
    """
    with open("./networks/" + nom, "rb") as f:
        net = pickle.load(f)
        f.close()
        print(f'Network {nom} loaded!')
        return net


#Fonction supplementaire
def delta_vecteur(j, size):
    """
    :return: np.array de taille (size, 1) avec un 1 a la position j, 0 partout ailleurs
    """
    e = np.zeros((size, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    print('This is a module, not a program')