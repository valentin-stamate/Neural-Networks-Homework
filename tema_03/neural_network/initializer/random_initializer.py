import numpy as np


class RandomInitializer:

    @staticmethod
    def init(n, m):
        return (1 - (-1)) * np.random.rand(n, m) + (-1)

