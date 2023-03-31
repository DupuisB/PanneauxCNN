import numpy as np

class Cost():
    pass

class QuadraticCost2(Cost):
    @staticmethod
    def fn(a, y):
        return np.mean(np.power(y - a, 2))

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class QuadraticCost(Cost):
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return np.linalg.norm(a - y)

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class CrossEntropyCost(Cost):
    """Mettre softmax en dernier"""
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def prime(a, y):
        """
        Return the gradient of the cost function with respect to the output ``a`` and desired output ``y``.
        """
        return (a - y) / (a * (1 - a))
