import numpy as np


class CrossEntropy:
    @staticmethod
    def der(error, out, act, shape, inp=None):
        der = np.zeros(shape, dtype='float32')

        for i in range(shape[0]):
            for j in range(shape[1]):
                der[i, j] = (error[i] * (-1))

        return der
