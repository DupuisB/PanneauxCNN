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
        return 2 * (a - y) / np.size(a)

class QuadraticCost(Cost):
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return np.linalg.norm(a - y) #(a - y, 'norm name')

    @staticmethod
    def delta(z, a, y, fn):
        """Return the error delta from the output layer."""
        return (a - y) * fn.prime(z)

    @staticmethod
    def prime(a, y):
        return 2 * (a - y) / np.size(y)

class CrossEntropyCost(Cost):
    def fn(self, a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def delta(self, z, a, y, fn):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a - y)
