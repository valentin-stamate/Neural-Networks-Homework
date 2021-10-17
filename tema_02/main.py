from dataset.dataset import Dataset
from neural_network.activation.binary import BinaryActivation
from neural_network.initializer.random_initializer import RandomInitializer
from neural_network.neural_network import NeuralNetwork


def main():
    dataset = Dataset('dataset_source/mnist.pkl.gz')

    nn = NeuralNetwork((28 * 28, 10), BinaryActivation(), RandomInitializer(), 0.01)

    nn.train(dataset, 10, 1000)
    nn.verify(dataset.test_set)
    nn.verify(dataset.valid_set)


if __name__ == '__main__':
    main()

