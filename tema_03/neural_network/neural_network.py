import numpy as np


class NeuralNetwork:

    def __init__(self, layers, activation, cost, initializer, learning_rate):
        if len(layers) != len(activation) + 1:
            print("Layers and activation functions should match")
            return

        self.batch_file = open('batch', 'a')
        self.epoch_file = open('epoch', 'a')

        self.activation = activation
        self.cost = cost
        self.brain = []
        self.biases = []
        self.learning_rate = learning_rate

        self.brain_buffer = []
        self.brain_buffer_tr = []
        self.biases_buffer = []

        for i in range(len(layers) - 1):
            n = layers[i + 1]
            m = layers[i]

            self.brain.append(initializer.init(n, m))
            self.biases.append(initializer.init(n, 1))

            self.brain_buffer.append(np.zeros((n, m), dtype='float32'))
            self.brain_buffer_tr.append(np.zeros((m, n), dtype='float32'))
            self.biases_buffer.append(np.zeros((n, 1), dtype='float32'))

    def feed_forward(self, input_):
        input_ = np.reshape(input_, (len(input_), 1))
        current_input = input_

        for i in range(len(self.brain)):
            current_input = self.theta(np.dot(self.brain[i], current_input) + self.biases[i], i)

        return current_input

    # demo, not generalized yet
    def train(self, dataset, epochs_len, batches_len, batch_limit=None):
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

                if batch_limit is not None:
                    if batch_limit <= batch_count:
                        break

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

                    for ind in range(len(self.brain) - 1, -1, -1):
                        der = self.cost[ind].der(errors[ind + 1], outputs[ind + 1],
                                                 self.activation[ind], self.brain[ind].shape, outputs[ind])

                        der_bias = self.cost[ind].der(errors[ind + 1],
                                                      outputs[ind + 1], self.activation[ind], self.biases[ind].shape)

                        delta_w[ind] += der
                        delta_b[ind] += der_bias

                for i in range(len(self.brain)):
                    drop = (1 - 0.0001)
                    self.brain[i] = self.brain[i] * drop - (delta_w[i] * self.learning_rate)
                    self.biases[i] = self.biases[i] * drop - (delta_b[i] * self.learning_rate)

                batch_error /= len(batch)
                batch_accuracy = int((1 - batch_error) * 10000) / 100
                print(f"Batch {batch_count}: progressive accuracy -> {batch_accuracy}%")

                self.batch_file.write(f'{batch_accuracy} ')
                self.batch_file.flush()

            epoch_error /= len(dataset.train_set)
            epoch_accuracy = int((1 - epoch_error) * 10000) / 100
            print(f"Epoch {e}: progressive accuracy -> {epoch_accuracy}%")
            self.epoch_file.write(f'{epoch_accuracy} ')
            self.epoch_file.flush()

        print("Training Finished")

    """ :param output (n, 1)
    """
    def theta(self, output, layer):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = self.activation[layer].fun(i + j, output)

        return output

    def get_outputs(self, inp):
        input_ = np.reshape(inp, (len(inp), 1))
        current_input = input_

        outputs = [current_input]
        for i in range(len(self.brain)):
            current_input = self.theta(np.dot(self.brain[i], current_input) + self.biases[i], i)
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
