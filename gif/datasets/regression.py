import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing, \
    load_diabetes, load_boston

from .utils import URLFile, split_set
from .full_dataset import FullDataset


class RegressionFullDataset(FullDataset):
    pass


class CTSlice(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 2000, 51500

    def load_(self):
        axv = URLFile("https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip",
                      folder=self.folder)
        raw = pd.read_csv(axv.access()).to_numpy()

        # with(zipfile.ZipFile(axv.access())) as zf:
        #     buffer = zf.read("slice_localization_data.csv")
        #     raw = pd.read_csv(zf).to_numpy()

        X = raw[:, :-1]
        y = raw[:, -1]

        split_set(self, X, y)


class Friedman1(RegressionFullDataset):
    def __init__(self, folder=None):
        super().__init__(folder)
        self.random_state = 42
        self.noise = 1

    @classmethod
    def get_default_lengths(cls):
        return 300, 2000

    def load_(self):
        X, y = make_friedman1(len(self), random_state=self.random_state,
                              noise=self.noise)
        split_set(self, X, y)



class Cadata(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 12384, 8256

    def load_(self):
        X, y = fetch_california_housing(data_home=self.folder, return_X_y=True)
        split_set(self, X, y)

class Abalone(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 2506, 1671

    def load_(self):
        column_names = ["sex", "length", "diameter", "height", "whole weight",
                    "shucked weight", "viscera weight", "shell weight", "rings"]

        dfile = URLFile("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
        data = pd.read_csv(dfile.access(), names=column_names)
        # Encode sex
        for label in "MFI":
            data[label] = data["sex"] == label
        del data["sex"]
        # Extract targets
        y = data.rings.values
        del data["rings"] # remove rings from data, so we can convert all the dataframe to a numpy 2D array.
        X = data.values.astype(np.float)

        split_set(self, X, y)



class OzoneLA(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 297, 33


    def load_(self):
        axv = URLFile("https://web.stanford.edu/~hastie/ElemStatLearn//datasets/LAozone.data",
                      folder=self.folder)
        raw = pd.read_csv(axv.access(), sep=",").to_numpy().astype("float")


        y = raw[:, 0]
        X = raw[:, 1:]

        split_set(self, X, y)


class Diabetes(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 398, 44

    def load_(self):
        X, y = load_diabetes(return_X_y=True)
        split_set(self, X, y)



class Hardware(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 188, 21


    def load_(self):
        axv = URLFile("https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
                      folder=self.folder)
        raw = pd.read_csv(axv.access(), sep=",", header=None,
                          usecols=list(range(2, 9))).to_numpy().astype("float")


        y = raw[:, -1]
        X = raw[:, :-1]

        split_set(self, X, y)



class BostonHousing(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 455, 51

    def load_(self):
        X, y = load_boston(return_X_y=True)
        split_set(self, X, y)



class MPG(RegressionFullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 353, 39


    def load_(self):
        axv = URLFile("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                      folder=self.folder)

        data = []
        with open(axv.access()) as hdl:
            for line in hdl:
                raw_data = line.split("\t")[0]  # remove model
                if "?" in raw_data:
                    continue
                data.append(raw_data.split())

        data = np.array(data, dtype="float")

        y = data[:, 0]
        X = data[:, 1:]

        split_set(self, X, y)
