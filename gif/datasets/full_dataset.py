from abc import ABCMeta, abstractmethod
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from gif.datasets.utils import data_folder


class FullDataset(object, metaclass=ABCMeta):
    @classmethod
    def get_default_lengths(cls):
        return 0, 0

    @classmethod
    def get_default_folder_name(cls):
        return cls.__name__.lower()

    def __init__(self, folder=None):
        if folder is None:
            folder = data_folder(self.__class__.get_default_folder_name())
        self.folder = folder
        self.tr_X_y = None
        self.ts_X_y = None

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def __len__(self):
        if self.ts_X_y is None:
            return sum(self.__class__.get_default_lengths())
        return len(self.tr_X_y[-1]) + len(self.ts_X_y[-1])

    def load_(self):
        pass

    def load(self):
        if self.tr_X_y is None:
            self.load_()

    def partition(self, train_size=None, shuffle=True, random_state=1217):
        self.load()
        if train_size is None:
            # Use default train size
            train_size = len(self.tr_X_y[-1])

        X_tr, y_tr = self.tr_X_y
        X_ts, y_ts = self.ts_X_y

        X = np.vstack((X_tr, X_ts))
        y = np.hstack((y_tr, y_ts))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,  train_size=train_size, shuffle=shuffle,
            random_state=random_state
        )

        self.tr_X_y = X_train, y_train
        self.ts_X_y = X_test, y_test



    @property
    def training_set(self):
        if self.tr_X_y is None:
            return np.array([]), np.array([])
        return self.tr_X_y

    @property
    def test_set(self):
        if self.ts_X_y is None:
            return np.array([]), np.array([])
        return self.ts_X_y

    def is_artificial(self):
        return hasattr(self, "random_state")
