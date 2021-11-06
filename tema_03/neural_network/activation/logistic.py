import math


class LogisticActivation:

    @staticmethod
    def fun(x):
        return 1 / (1 + (math.e ** (-x)))

    @staticmethod
    def der(x):
        y = LogisticActivation.fun(x)
        return y * (1 - y)
