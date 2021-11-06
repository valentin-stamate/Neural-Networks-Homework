import math


class TanhActivation:
    @staticmethod
    def fun(x):
        e = math.e
        return ((e ** x) - (e ** (-x))) * ((e ** x) + (e ** (-x)))

    @staticmethod
    def der(x):
        return 1 - TanhActivation.fun(x) ** 2
