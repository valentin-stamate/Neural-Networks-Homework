import math


class SoftMaxActivation:

    @staticmethod
    def fun(i, values):
        x = values[i]

        su = 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                su += (math.e ** (values[i, j]))

        return (math.e ** x) / su


