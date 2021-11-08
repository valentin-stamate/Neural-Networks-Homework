from dataset.dataset import Dataset
from neural_network.activation.logistic import LogisticActivation
from neural_network.activation.softman import SoftMaxActivation
from neural_network.cost.crossentropy import CrossEntropy
from neural_network.cost.quadratic_cost import QuadraticCost
from neural_network.initializer.std_initilaizer import StandardDeviationInitializer
from neural_network.neural_network import NeuralNetwork


def main():
    dataset = Dataset('dataset_source/mnist.pkl.gz')

    log = LogisticActivation()
    softmax = SoftMaxActivation()

    nn = NeuralNetwork((28 * 28, 100, 10), [log, softmax], [QuadraticCost(), CrossEntropy()],
                       StandardDeviationInitializer(), 0.01)

    nn.train(dataset, 10, 1000)
    nn.verify(dataset.test_set)

    print("Done")


if __name__ == '__main__':
    main()
