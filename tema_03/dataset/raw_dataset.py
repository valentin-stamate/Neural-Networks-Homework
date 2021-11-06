import _pickle as pickle
import gzip


class RawDataset:

    @staticmethod
    def process(path):
        with gzip.open(path, 'rb') as fd:
            train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
            return train_set, valid_set, test_set
