import numpy as np
from scipy import signal
from utils.activation_functions import *
import time

# @ = np.dot, * = np.multiply (hadamart)

class Layer():
    def backpropagation_update(self, eta, batch_size):
        pass

    def reset_backpropagation(self):
        pass


class Dense(Layer):
    def __init__(self, entree_dim, sortie_dim, func):
        self.entree_dim, self.dim_sortie = entree_dim, sortie_dim
        self.func = func
        self.poids = np.random.randn(sortie_dim, entree_dim)
        self.biais = np.random.randn(sortie_dim, 1)
        self.poids_batch = np.zeros(self.poids.shape)
        self.biais_batch = np.zeros(self.biais.shape)

    def feedforward(self, entree):
        self.entree = entree
        self.sortie = self.func.fn(self.poids @ self.entree + self.biais)
        return self.sortie

    def backpropagation(self, eta, nabla_sortie):  # reset indique fin du batch
        #st = time.time()
        derivee_activation = self.func.prime(self.sortie)
        nabla_sortie = nabla_sortie * derivee_activation  # fait la derivee de la fct d'activation
        nabla_poids = (nabla_sortie @ self.entree.T)
        nabla_entree = (self.poids.T @ nabla_sortie)  # Jacobienne
        # self.poids -= eta * nabla_poids
        # self.biais -= eta * nabla_sortie
        self.poids_batch += eta * nabla_poids
        self.biais_batch += eta * nabla_sortie
        #print(time.time() - st)
        #print('dens\n')
        return nabla_entree

    def backpropagation_update(self, eta, batch_size):
        self.poids -= eta * self.poids_batch / batch_size
        self.biais -= eta * self.biais_batch / batch_size

    def reset_backpropagation(self):
        self.poids_batch = np.zeros(self.poids.shape)
        self.biais_batch = np.zeros(self.biais.shape)

    def name(self):
        return f"Dense({self.entree_dim}, {self.dim_sortie}, {self.func.__name__})"


class Convolution(Layer):
    """stride = 1, padding = 0, pas de pooling"""
    def __init__(self, img_dim, kernel_cote, sortie_prof, func):
        self.func = func
        self.entree_dims = img_dim
        self.entree_haut, self.entree_larg, self.entree_prof = self.entree_dims
        self.sortie_dims = (self.entree_haut - kernel_cote + 1, self.entree_larg - kernel_cote + 1, sortie_prof)
        self.sortie_prof = sortie_prof
        self.filtre_dim = (kernel_cote, kernel_cote, self.entree_prof, sortie_prof)
        self.filtre = np.random.rand(*self.filtre_dim)
        self.biais = np.random.rand(*self.sortie_dims)
        self.filtre_batch = np.zeros(self.filtre.shape)
        self.biais_batch = np.zeros(self.biais.shape)

    def feedforward(self, entree):
        self.entree = entree
        self.sortie = np.copy(self.biais)
        for i in range(self.sortie_prof):
            for j in range(self.entree_prof):
                self.sortie[:, :, i] += signal.correlate2d(self.entree[:, :, j], self.filtre[:, :, j, i], "valid")
                # self.sortie[i] += np.convolve(self.entree[j], self.filtre[i, j], mode='full')
                # self.sortie[i] += np.real(np.ifft2(np.fft2(self.entree[j])*np.fft2(self.filtre[i, j], s=self.entree[j].shape)))
        self.sortie = self.func.fn(self.sortie)
        return self.sortie

    def backpropagation(self, eta, nabla_sortie):
        #st = time.time()
        derivee_activation = self.func.prime(self.sortie)
        nabla_sortie = nabla_sortie * derivee_activation
        nabla_filtre = np.zeros(self.filtre_dim)
        nabla_entree = np.zeros(self.entree_dims)

        for i in range(self.sortie_prof):
            for j in range(self.entree_prof):
                nabla_filtre[:, :, j, i] = signal.correlate2d(self.entree[:, :, j], nabla_sortie[:, :, i], "valid")
                nabla_entree[:, :, j] += signal.convolve2d(nabla_sortie[:, :, i], self.filtre[:, :, j, i], "full")
                # nabla_filtre[i,j] = np.convolve(self.entree[j], nabla_sortie[i], mode='valid')
                # nabla_entree[j] += np.convolve(nabla_sortie[i], self.filtre[i], mode='full')
        self.filtre_batch += nabla_filtre
        self.biais_batch += nabla_sortie
        #print(time.time() - st)
        #print('conv\n')
        return nabla_entree

    def name(self):
        return f"Convolution({self.entree_dims}, {self.filtre_dim[0]}, {self.sortie_prof}, {self.func.__name__})"

    def backpropagation_update(self, eta, batch_size):
        self.filtre -= eta * self.filtre_batch / batch_size
        self.biais -= eta * self.biais_batch / batch_size

    def reset_backpropagation(self):
        self.filtre_batch = np.zeros(self.filtre.shape)
        self.biais_batch = np.zeros(self.biais.shape)


class Reshape(Layer):
    def __init__(self, dim_entree, dim_sortie):
        self.dim_entree = dim_entree
        self.dim_sortie = dim_sortie

    def feedforward(self, entree):
        self.entree = entree
        temp = np.reshape(entree, self.dim_sortie)
        return temp

    def backpropagation(self, eta, nabla_sortie):
        return np.reshape(nabla_sortie, self.dim_entree)

    def name(self):
        return f"Reshape({self.dim_entree}, {self.dim_sortie})"


class Softmax(Layer):
    """Normalise pour avoir somme des sorties = 1"""
    def __init__(self):
        pass

    def feedforward(self, entree):
        expo = np.exp(entree - np.max(entree))
        self.sortie = expo / np.sum(expo)
        return self.sortie

    def backpropagation(self, eta, nabla_sortie):
        jacobian = np.diag(self.sortie.flatten()) - np.outer(self.sortie, self.sortie)
        nabla_entree = jacobian @ nabla_sortie
        return nabla_entree

    def name(self):
        return "Softmax"



class Maxpooling(Layer):
    """filtre carre, cote*cote, aucun parametre (donc rien a update en backprop), filtre_cote divise la hauteur ET la largeur"""

    def __init__(self, entree_dim, filtre_cote):
        self.entree_haut, self.entree_larg, self.entree_prof = entree_dim
        self.entree_dim = entree_dim
        self.filtre_cote = filtre_cote
        self.sortie_haut = self.entree_haut // self.filtre_cote
        self.sortie_larg = self.entree_larg // self.filtre_cote
        self.sortie_prof = self.entree_prof

    def feedforward(self, entree):
        self.entree = entree
        cote = self.filtre_cote
        sortie = np.zeros((self.sortie_haut, self.sortie_larg, self.sortie_prof))
        for couche in range(self.sortie_prof):
            for y in range(self.sortie_haut):
                for x in range(self.sortie_larg):
                    sortie[x, y, couche] = np.max(entree[y * cote: (y + 1) * cote, x * cote: (x + 1) * cote, couche])
        return sortie

    def backpropagation(self, grad_sortie, eta):
        grad_entree = np.zeros((self.entree_haut, self.entree_larg, self.entree_prof))
        entree = self.entree
        cote = self.filtre_cote
        for couche in range(self.sortie_prof):
            for x in range(self.sortie_larg):  # j
                for y in range(self.sortie_haut):
                    origine = entree[x * cote: (x + 1) * cote, y * cote: (y + 1) * cote, couche]
                    x_max, y_max = np.where(np.max(origine) == origine)
                    x_max, y_max = x_max[0], y_max[0]
                    grad_entree[x * cote: (x + 1) * cote, y * cote: (y + 1) * cote, couche][x_max, y_max] = grad_sortie[
                        x, y, couche]
        return grad_entree

    def name(self):
        return f"Maxpooling({self.entree_dim}, {self.filtre_cote})"


class Greyed(Layer):
    """Convertit l'image en niveaux de gris"""
    def __init__(self, first = True):
        self.rgb_weights = np.array([0.2989, 0.5870, 0.1140])
        self.first = first

    def feedforward(self, entree):
        self.entree = entree
        return (entree[..., :3] @ self.rgb_weights)[..., None]  # [..., None] pour ajouter la dimension manquante

    def backpropagation(self, nabla_sortie, eta):
        if self.first:
            return nabla_sortie
        nabla_entree = np.zeros_like(self.entree)
        nabla_entree[..., :3] = (nabla_sortie[..., 0, None] @ self.rgb_weights[None, :])
        return nabla_entree

    def name(self):
        return "Greyed"

