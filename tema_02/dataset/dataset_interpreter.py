from matplotlib import pyplot as plt
import numpy as np


class DatasetInterpreter:

    @staticmethod
    def show_as_image(array, n):
        array = np.reshape(array, (n, n))
        plt.imshow(array, interpolation='nearest', cmap='gray')
        plt.show()
