import numpy as np
from scipy import signal
from activation_functions import *

#@ = np.dot, * = np.mltiply (hadamart)

class Layer():
    def backpropagation_update(self, eta, batch_size):
        pass

    def reset_backpropagation(self):
        pass

    pass


class Dense(Layer):
    def __init__(self, dim_entree, dim_sortie, func):
        self.dim_entree, self.dim_sortie = dim_entree, dim_sortie
        self.func = func
        self.poids = np.random.randn(dim_sortie, dim_entree)
        self.biais = np.random.randn(dim_sortie, 1)
        self.poids_batch = np.zeros(self.poids.shape)
        self.biais_batch = np.zeros(self.biais.shape)

    def feedforward(self, entree):
        self.entree = entree
        self.sortie = self.func.fn(self.poids @ self.entree + self.biais)
        return self.sortie

    def backpropagation(self, eta, nabla_sortie):  # reset indique fin du batch
        derivee_activation = self.func.prime(self.sortie)  # % fonction d'activation je remet self.Sortie pcq autcun sens mais amrchait ?
        nabla_sortie = nabla_sortie * derivee_activation
        nabla_poids = (nabla_sortie @ self.entree.T)
        nabla_entree = (self.poids.T @ nabla_sortie) # Jacobienne
        # self.poids -= eta * nabla_poids
        # self.biais -= eta * nabla_sortie
        self.poids_batch += nabla_poids
        self.biais_batch += nabla_sortie
        return nabla_entree


    def backpropagation_update(self, eta, batch_size):
        self.poids -= eta * self.poids_batch / batch_size
        self.biais -= eta * self.biais_batch / batch_size

    def reset_backpropagation(self):
        self.poids_batch = np.zeros(self.poids.shape)
        self.biais_batch = np.zeros(self.biais.shape)

    def name(self):
        return f"Dense({self.dim_entree}, {self.dim_sortie}, {self.func.__name__})"


class Convolution(Layer):

    def __init__(self, img_dims, kernel_cote, sortie_prof, func):
        self.func = func
        self.entree_dims = img_dims
        self.entree_haut, self.entree_larg, self.entree_prof = self.entree_dims
        self.sortie_dims = (self.entree_haut - kernel_cote + 1, self.entree_larg - kernel_cote + 1, sortie_prof)
        self.sortie_prof = sortie_prof
        self.filtre_dims = (kernel_cote, kernel_cote, self.entree_prof, sortie_prof)
        self.filtre = np.random.rand(*self.filtre_dims)
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
        derivee_activation = self.func.prime(self.sortie)  # % fonction d'activation je remet self.Sortie pcq autcun sens mais amrchait ?
        nabla_sortie = nabla_sortie * derivee_activation
        nabla_filtre = np.zeros(self.filtre_dims)
        nabla_entree = np.zeros(self.entree_dims)

        for i in range(self.sortie_prof):
            for j in range(self.entree_prof):
                nabla_filtre[:, :, j, i] = signal.correlate2d(self.entree[:, :, j], nabla_sortie[:, :, i], "valid")
                nabla_entree[:, :, j] += signal.convolve2d(nabla_sortie[:, :, i], self.filtre[:, :, j, i], "full")
                #nabla_filtre[i,j] = np.convolve(self.entree[j], nabla_sortie[i], mode='valid')
                #nabla_entree[j] += np.convolve(nabla_sortie[i], self.filtre[i], mode='full')
        self.filtre_batch += nabla_filtre
        self.biais_batch += nabla_sortie
        return nabla_entree

    def name(self):
        return f"Convolution({self.entree_dims}, {self.filtre_dims[0]}, {self.sortie_prof}, {self.func.__name__})"

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
    def __init__(self):
        pass

    def feedforward(self, entree):
        maxi = np.max(entree)
        entree = entree - maxi
        expo = np.exp(entree)
        self.sortie = expo / np.sum(expo)
        return self.sortie

    def backpropagation(self, grad_sortie, eta):
        return grad_sortie

    def name(self):
        return "Softmax"


class Greyed(Layer):
    """Seulement en premiere couche ! (pas le bon gradient)"""
    def __init__(self):
        pass

    def feedforward(self, entree):
        return (entree[...,:3] @ [0.2989, 0.5870, 0.1140])[..., None] #[..., None] pour le format la dim manquante

    def backpropagation(self, grad_sortie, eta):
        return grad_sortie

    def name(self):
        return "Greyed"
