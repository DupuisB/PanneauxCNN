import numpy as np

class Cost():
    pass

class DistanceEuclidienne2(Cost):
    """
    Renvoie le carré de la distance euclidienne entre a et y
    """
    @staticmethod
    def fn(a, y):
        return np.mean(np.power(y - a, 2))

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class QuadraticCost2(Cost):
    """
    Renvoie le carré de la distance euclidienne entre a et y
    """
    @staticmethod
    def fn(a, y):
        return np.mean(np.power(y - a, 2))

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class DistanceEuclidienne(Cost):
    """
    Renvoie la distance euclidienne entre a et y
    """
    @staticmethod
    def fn(a, y):
        return np.linalg.norm(a - y)

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class EntropieCroisee(Cost):
    """
    Mettre softmax en dernier (distribution de probas)
    """
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def prime(a, y):
        return (a - y) / (a * (1 - a))
