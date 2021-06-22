import gzip
import pickle
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification, \
    make_hastie_10_2, fetch_covtype

from .utils import URLFile, split_set, openml_fetch_set

from .full_dataset import FullDataset


class ClassificationFullDataset(FullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 0

    @classmethod
    def is_binary_classification(cls):
        return cls.get_default_n_classes() == 2

    @property
    def n_classes(self):
        return self.__class__.get_default_n_classes()

    @property
    def n_classes(self):
        return self.__class__.get_default_n_classes()


# ====================================================================== OPEN ML

class Waveform(ClassificationFullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 3

    @classmethod
    def get_default_lengths(cls):
        return 3500, 1500

    def load_(self):
        openml_fetch_set(self, 60)


class Twonorm(ClassificationFullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 300, 7100

    def load_(self):
        openml_fetch_set(self, 1507)

class Ringnorm(ClassificationFullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 300, 7100

    def load_(self):
        openml_fetch_set(self, 1496)


class Musk2(ClassificationFullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 2000, 4598

    def load_(self):
        bunch = fetch_openml(data_id=1116, data_home=self.folder)
        y = bunch.target.to_numpy().astype(int)
        X = bunch.data
        X = X.iloc[:, 1:].to_numpy()

        # All the musk are at the start and the non-musk after: interleave it
        X_min = np.zeros(X.shape, X.dtype)
        y_mix = np.zeros(y.shape, y.dtype)

        X_min[0:len(y):2] = X[:len(y) // 2]
        X_min[1:len(y):2] = X[len(y) // 2:]
        y_mix[0:len(y):2] = y[:len(y) // 2]
        y_mix[1:len(y):2] = y[len(y) // 2:]

        split_set(self, X_min, y_mix)




class Vowel(ClassificationFullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 11
    @classmethod
    def get_default_lengths(cls):
        return 495, 495

    def load_(self):
        bunch = fetch_openml(data_id=307, data_home=self.folder)
        y = np.zeros(len(bunch.target), dtype=int)
        labels = np.unique(bunch.target)
        for i, vowel in enumerate(labels):
            y[bunch.target == vowel] = i
        X = bunch.data.iloc[:, 1:].to_numpy()

        split_set(self, X, y)

class BinaryVowel(Vowel):
    @classmethod
    def get_default_folder_name(cls):
        return "vowel"

    @classmethod
    def get_default_n_classes(cls):
        return 2

    def load_(self):
        super().load_()
        _, y = self.tr_X_y
        y[:] = (y < 6).astype(int)
        _, y = self.ts_X_y
        y[:] = (y < 6).astype(int)



# ==================================================================== GENERATED
class Madelon(ClassificationFullDataset):
    def __init__(self, folder=None):
        super(Madelon, self).__init__(folder)
        self.random_state = 42

    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 4400, 4400

    def load_(self):
        X, y = make_classification(n_samples=len(self), n_features=500,
                                   n_informative=20, n_redundant=50,
                                   random_state=self.random_state)

        split_set(self, X, y)



class Hastie(ClassificationFullDataset):
    def __init__(self, folder=None):
        super().__init__(folder)
        self.random_state = 42

    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 2000, 10000

    def load_(self):
        X, y = make_hastie_10_2(random_state=self.random_state)
        y = (y + 1) / 2.
        y = y.astype(int)
        split_set(self, X, y)


# ================================================================ OTHER SKLEARN
class Covertype(ClassificationFullDataset):
    @classmethod
    def get_default_n_classes(cls):
        return 7

    @classmethod
    def get_default_lengths(cls):
        return 348607, 232405

    def load_(self):
        X, y = fetch_covtype(data_home=self.folder, return_X_y=True)
        split_set(self, X, y)

class BinaryCovertype(Covertype):
    @classmethod
    def get_default_folder_name(cls):
        return "covertype"

    @classmethod
    def get_default_n_classes(cls):
        return 2

    def load_(self):
        X, y = fetch_covtype(data_home=self.folder, return_X_y=True)
        y[y == 1] = 0
        y[y == 4] = 0
        y[y == 5] = 0
        y[y == 6] = 0
        y[y == 7] = 0
        y[y == 2] = 1
        y[y == 3] = 1
        split_set(self, X, y)


# ======================================================================= CUSTOM
class Letters(ClassificationFullDataset):
    @classmethod
    def get_default_folder_name(cls):
        return "letters"

    @classmethod
    def get_default_n_classes(cls):
        return 26

    @classmethod
    def get_default_lengths(cls):
        return 16000, 4000

    def load_(self):
        dfile = URLFile("https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
                        folder=self.folder)
        df = pd.read_csv(dfile.access(), header=None)
        alpha2cls = {chr(ord("A") + n): n for n in range(26)}
        y = np.array([alpha2cls[x] for x in df.iloc[:, 0]], dtype=np.int64)
        X = df.iloc[:, 1:].to_numpy().astype(np.float64)

        split_set(self, X, y)


class BinaryLetters(Letters):
    @classmethod
    def get_default_n_classes(cls):
        return 2

    def load_(self):
        super().load_()

        X, y = self.tr_X_y
        y[:] = (y < 13).astype(int)

        X, y = self.ts_X_y
        y[:] = (y < 13).astype(int)



class MNIST(ClassificationFullDataset):
    @classmethod
    def get_default_folder_name(cls):
        return "mnist"

    @classmethod
    def get_default_n_classes(cls):
        return 10

    @classmethod
    def get_default_lengths(cls):
        return 60000, 10000

    def load_(self):
        axv = URLFile("http://www.montefiore.ulg.ac.be/~jmbegon/permanent/mnist.pkl.gz",
                      folder=self.folder)

        with gzip.open(axv.access(), "rb") as f:
            (X_t, y_t), (X_v, y_v), (X_ts, y_ts) = pickle.load(f, encoding='latin1')

        X_t = np.vstack((X_t, X_v))
        y_t = np.hstack((y_t, y_v))

        self.tr_X_y = X_t, y_t
        self.ts_X_y = X_ts, y_ts


class MNIST8vs9(MNIST):
    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 11800, 1983

    def load_(self):
        super().load_()
        X, y = self.tr_X_y
        sel = np.logical_or(y == 8, y == 9)
        self.tr_X_y =  X[sel], y[sel]

        X, y = self.ts_X_y
        sel = np.logical_or(y == 8, y == 9)
        self.ts_X_y = X[sel], y[sel]


class BinaryMNIST(MNIST):

    @classmethod
    def get_default_n_classes(cls):
        return 2

    @classmethod
    def get_default_lengths(cls):
        return 60000, 10000

    def load_(self):
        super().load_()
        _, y = self.tr_X_y
        y[:] = (y < 5).astype(int)

        _, y = self.ts_X_y
        y[:] = (y < 5).astype(int)

