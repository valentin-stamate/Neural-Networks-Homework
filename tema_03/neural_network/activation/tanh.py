import math


class TanhActivation:
    @staticmethod
    def fun(i, values):
        e = math.e
        x = values[i]
        return ((e ** x) - (e ** (-x))) * ((e ** x) + (e ** (-x)))

    @staticmethod
    def der(i, values):
        return 1 - TanhActivation.fun(i, values) ** 2
