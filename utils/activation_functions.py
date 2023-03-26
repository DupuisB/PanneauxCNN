import numpy as np
class activations:
    """Fonction d'activation, (fn, prime) possible"""
    pass

class sigmoid(activations):
    @staticmethod
    def fn(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def prime(z):
        return sigmoid.fn(z)*(1-sigmoid.fn(z))

class tanh(activations):
    @staticmethod
    def fn(z):
        return (np.tanh(z) + 1)/2
    @staticmethod
    def prime(z):
        return (1-np.tanh(z)**2)/2

class ReLU(activations):
    @staticmethod
    def fn(z):
        return np.maximum(0, z)
    @staticmethod
    def prime(z):
        return np.where(z > 0, 1, 0)

class identity(activations):
    @staticmethod
    def fn(z):
        return z
    @staticmethod
    def prime(z):
        return 1
