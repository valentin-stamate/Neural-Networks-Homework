import math


class LogisticActivation:

    @staticmethod
    def fun(i, values):
        x = values[i]
        return 1 / (1 + (math.e ** (-x)))

    @staticmethod
    def der(i, values, is_output=False):
        x = values[i]
        y = LogisticActivation.fun(i, values)

        if is_output:
            y = x

        return y * (1 - y)
