import numpy as np


class NeuralNetwork:

    def __init__(self, layers, activation, initializer, learning_rate):
        self.activation = activation
        self.brain = []
        self.biases = []
        self.learning_rate = learning_rate

        for i in range(len(layers) - 1):
            self.brain.append(initializer.init(layers[i + 1], layers[i]))
            self.biases.append(initializer.init(layers[i + 1], 1))

    def feed_forward(self, input_):
        input_ = np.reshape(input_, (len(input_), 1))
        current_input = input_

        for i in range(len(self.brain)):
            current_input = self.theta(np.dot(self.brain[i], current_input) + self.biases[i])

        return current_input

    # demo, not generalized yet
    def train(self, dataset, epochs_len, batches_len):
        print("Neural Network Training...")

        for e in range(epochs_len):
            dataset.shuffle()
            batches = dataset.get_batches(batches_len)

            error = 0
            for batch in batches:
                delta_w = np.zeros(self.brain[0].shape, dtype='float32')
                delta_b = np.zeros(self.biases[0].shape, dtype='float32')

                for instance in batch:
                    input_ = instance[0]
                    output = self.theta(np.dot(self.brain[0], input_) + self.biases[0])

                    error += NeuralNetwork.supervised_error(instance[1], output)

                    delta_w += self.learning_rate * np.dot((instance[1] - output), instance[2])
                    delta_b += self.learning_rate * (instance[1] - output)

                self.brain[0] = self.brain[0] + delta_w
                self.biases[0] = self.biases[0] + delta_b

            error /= len(dataset.train_set)
            accuracy = int((1 - error) * 10000) / 100
            print(f"Epoch {e}: progressive accuracy -> {accuracy}%")

        print("Training Finished")

    def theta(self, output):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = self.activation.fun(output[i, j])

        return output

    def verify(self, _set):
        print("Verifying the model...")
        errors = 0
        for instance in _set:
            output = self.feed_forward(instance[0])
            errors += self.supervised_error(instance[1], output)

        error = errors / len(_set)
        accuracy = int((1 - error) * 10000) / 100
        print(f"Accuracy -> {accuracy}%")

    @staticmethod
    def supervised_error(target, output):
        target_label = target.argmax()
        output_label = output.argmax()

        if target_label != output_label:
            return 1

        return 0

    def show_brain(self):
        for array in self.brain:
            print(array)

    def show_biases(self):
        for bias in self.biases:
            print(bias)