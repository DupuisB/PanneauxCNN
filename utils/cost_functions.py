import numpy as np


#### Define the quadratic and cross-entropy cost functions
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
        return np.linalg.norm(a - y) #(a - y, 'norm name')

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class CrossEntropyCost(Cost):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def prime(a, y):
        a = np.clip(a, 1e-7, 1 - 1e-7)
        y = np.clip(y, 1e-7, 1 - 1e-7)
        return ((1 - y) / (1 - a) - y / a) / np.size(y)