import numpy as np

from dataset.dataset import Dataset
from neural_network.activation.logistic import LogisticActivation
from neural_network.cost.quadratic_cost import QuadraticCost
from neural_network.initializer.random_initializer import RandomInitializer
from neural_network.neural_network import NeuralNetwork


def main():
    dataset = Dataset('dataset_source/mnist.pkl.gz')

    nn = NeuralNetwork((28 * 28, 100, 10), LogisticActivation(), RandomInitializer(), 0.01)

    nn.train(dataset, 10, 1000)
    # nn.verify(dataset.test_set)
    # nn.verify(dataset.valid_set)

    # a = np.array([1.0, 4.0, 3.0, 4.0])
    # b = np.array([2.0, 1.0, 0.0, 2.0])
    #
    # print(QuadraticCost.fun(a, b))

    # print(NeuralNetwork.supervised_error([0.01, 0.01, 0.02, 0], [0, 0, 1, 0]))

    # nn = NeuralNetwork((3, 2), LogisticActivation(), RandomInitializer(), 0.01)
    # outputs = nn.get_outputs(np.array([0.1, 0.3, 0.01]))

    # nn.show_brain()
    # nn.show_biases()
    # print()
    #
    # for out in outputs:
    #     print(out)
    #     print()

    print("Done")

if __name__ == '__main__':
    main()

