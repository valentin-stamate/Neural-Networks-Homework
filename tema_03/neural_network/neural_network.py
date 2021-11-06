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
            epoch_error = 0

            dataset.shuffle()
            batches = dataset.get_batches(batches_len)

            batch_count = 0
            for batch in batches:
                batch_error = 0
                delta_w = []
                delta_b = []

                batch_count += 1

                for i in range(len(self.brain)):
                    delta_w.append(np.zeros(self.brain[i].shape, dtype='float32'))
                    delta_b.append(np.zeros(self.biases[i].shape, dtype='float32'))

                for instance in batch:
                    input_ = instance[0]
                    target_ = instance[1]

                    outputs = self.get_outputs(input_)
                    errors = self.get_errors(target_ - outputs[-1])

                    err = NeuralNetwork.supervised_error(instance[1], outputs[-1])
                    batch_error += err
                    epoch_error += err

                    for l in range(len(self.brain) - 1, -1, -1):
                        # derivative = [(ti - oi)(-1)][oi(1 - oi)][ij]

                        der = self.get_der(l, outputs, errors, self.brain[l].shape)
                        der_bias = self.get_der(l, outputs, errors, self.biases[l].shape, bias=True)

                        delta_w[l] += der
                        delta_b[l] += der_bias

                for i in range(len(self.brain)):
                    self.brain[i] = self.brain[i] - (delta_w[i] * self.learning_rate)
                    self.biases[i] = self.biases[i] - (delta_b[i] * self.learning_rate)

                batch_error /= len(batch)
                batch_accuracy = int((1 - batch_error) * 10000) / 100
                print(f"Batch {batch_count}: progressive accuracy -> {batch_accuracy}%")

            epoch_error /= len(dataset.train_set)
            epoch_accuracy = int((1 - epoch_error) * 10000) / 100
            print(f"Epoch {e}: progressive accuracy -> {epoch_accuracy}%")

        print("Training Finished")

    def get_der(self, weights_index, outputs, errors, shape, bias=False):
        al = np.zeros(shape, dtype='float32')

        for i in range(shape[0]):
            for j in range(shape[1]):
                index = weights_index + 1
                val = 1

                if not bias:
                    val = outputs[index - 1][j]

                al[i, j] = errors[index][i] * (-1) * outputs[index][i] * (1 - outputs[index][i]) * val

        return al

    def theta(self, output):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = self.activation.fun(output[i, j])

        return output

    def get_outputs(self, inp):
        input_ = np.reshape(inp, (len(inp), 1))
        current_input = input_

        outputs = [current_input]
        for i in range(len(self.brain)):
            current_input = self.theta(np.dot(self.brain[i], current_input) + self.biases[i])
            outputs.append(current_input)

        return outputs

    def get_errors(self, err):
        current_error = err
        errors = [current_error]

        for i in range(len(self.brain) - 1, -1, -1):
            trans = np.transpose(self.brain[i])
            current_error = np.dot(trans, current_error)
            errors.insert(0, current_error)

        return errors

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
        print("Brain:")
        for array in self.brain:
            print(array)

    def show_biases(self):
        print("Biases")
        for bias in self.biases:
            print(bias)
