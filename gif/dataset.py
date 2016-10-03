"""
A module for loading datasets
"""

# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# Licence: BSD 3 clause
import numpy as np

from sklearn.datasets.base import Bunch
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_hastie_10_2
from sklearn.utils import check_random_state


def partition_data(bunch):
    X, y = bunch.data, bunch.target
    ls_size = bunch.ls_size
    X_ls, y_ls = X[:ls_size], y[:ls_size]
    X_ts, y_ts = X[ls_size:], y[ls_size:]
    return X_ls, y_ls, X_ts, y_ts

def load_hastie(random_state=0):
    X,y = make_hastie_10_2(random_state=random_state)
    y = (y+1)/2.
    y = y.astype(long)
    ls_size = 2000
    return Bunch(data=X, target=y, ls_size=ls_size)

def load_friedman1(noise=1, random_state=0):
    X, y = make_friedman1(2300, random_state=random_state, noise=noise)
    ls_size = 300
    return Bunch(data=X, target=y, ls_size=ls_size)

def load_waveform(random_state=0):
    X, y = make_waveforms(5000, random_state=random_state)
    ls_size = 3500
    return Bunch(data=X, target=y, ls_size=ls_size)


def make_waveforms(n_samples=300, random_state=None):
    """Make the waveforms dataset. (CART)"""
    random_state = check_random_state(random_state)

    def h1(x):
        if x < 7:
            return x
        elif x < 13:
            return 13.-x
        else:
            return 0.

    def h2(x):
        if x < 9:
            return 0.
        elif x < 15:
            return x-9.
        else:
            return 21.-x

    def h3(x):
        if x < 5:
            return 0.
        elif x < 11:
            return x-5.
        elif x < 17:
            return 17.-x
        else:
            return 0.

    u = random_state.rand(n_samples)
    y = random_state.randint(low=0, high=3, size=n_samples)
    X = random_state.normal(size=(n_samples, 21))

    for i in range(n_samples):
        if y[i] == 0:
            ha = h1
            hb = h2
        elif y[i] == 1:
            ha = h1
            hb = h3
        else:
            ha = h2
            hb = h3

        for m in np.arange(1, 21+1):
            X[i, m-1] += u[i] * ha(m) + (1 - u[i]) * hb(m)

    return X, y
