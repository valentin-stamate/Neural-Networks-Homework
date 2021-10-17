import random

from dataset.raw_dataset import RawDataset
import numpy as np


class Dataset:

    def __init__(self, path):
        train_set, valid_set, test_set = RawDataset.process(path)

        self.train_set = Dataset.convert(train_set)
        self.valid_set = Dataset.convert(valid_set)
        self.test_set = Dataset.convert(test_set)

    def shuffle(self):
        self.train_set = random.sample(self.train_set, len(self.train_set))

    def get_batches(self, n):
        batch_len = len(self.train_set) // n

        batches = []
        for i in range(n):
            batch = []
            for j in range(i * batch_len, (i + 1) * batch_len):
                batch.append(self.train_set[j])
            batches.append(batch)

        return batches

    @staticmethod
    def get_random_perm(n):
        perm = [x for x in range(n)]
        perm = random.sample(perm, n)
        return perm

    def get_ts(self, index):
        return self.train_set[index][0], self.train_set[index][1]

    def get_vs(self, index):
        return self.valid_set[index][0], self.valid_set[index][1]

    def get_test_s(self, index):
        return self.test_set[index][0], self.test_set[index][1]

    def get_ts_len(self):
        return len(self.train_set[0])

    def get_vs_len(self):
        return len(self.valid_set[0])

    def get_test_s_len(self):
        return len(self.test_set[0])

    @staticmethod
    def convert(dt_set):
        c_set = []

        for i in range(len(dt_set[0])):
            input_ = dt_set[0][i]

            label_ = np.zeros((10, 1), dtype='float32')
            label_[dt_set[1][i]] = 1

            c_set.append([np.array(input_).flatten().reshape((28 * 28, 1)), label_, np.array(input_).flatten().reshape((1, 28 * 28))])

        return c_set
