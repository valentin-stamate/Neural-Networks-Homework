import numpy as np


class QuadraticCost:
    @staticmethod
    def fun(t, o):
        return ((t - o) ** 2) / 2.0

    @staticmethod
    def der(error, out, act, shape, inp=None):
        der = np.zeros(shape, dtype='float32')

        for i in range(shape[0]):
            for j in range(shape[1]):
                der[i, j] = (error[i] * (-1)) * act.der(i, out, is_output=True)

                if inp is not None:
                    der[i, j] *= inp[j]

        return der

