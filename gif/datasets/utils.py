import os
import shutil
import sys
from urllib.request import urlopen
from urllib.parse import urlparse

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def data_folder(subfolder=None):
    folder = os.environ.get("GIF_DATASET_FOLDER",
                            os.path.expanduser("~/data"))
    if subfolder is not None:
        folder = os.path.join(folder, subfolder)
    return folder


def split_set(receiver, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=receiver.__class__.get_default_lengths()[0],
        shuffle=False
    )

    receiver.tr_X_y = X_train, y_train
    receiver.ts_X_y = X_test, y_test


def openml_fetch_set(receiver, id):
    bunch = fetch_openml(data_id=id, data_home=receiver.folder)
    y = bunch.target.astype(int)
    X = bunch.data

    split_set(receiver, X, y)


class URLFile(object):
    def __init__(self, url, folder=None, fname=None, override=False,
                 output=sys.stdout):
        self.url = url
        if folder is None:
            folder = data_folder()
        self.folder = folder
        if fname is None:
            fname = os.path.basename(urlparse(url).path)
        self.fname = fname
        self.override = override
        self.output = output

    def __repr__(self):
        return "{}({}, {}, {}, {}, {})" \
               "".format(self.__class__.__name__,
                         repr(self.url),
                         repr(self.folder),
                         repr(self.fname),
                         repr(self.override),
                         repr(self.output))

    @property
    def fpath(self):
        return os.path.join(self.folder, self.fname)

    def download(self):
        os.makedirs(self.folder, exist_ok=True)
        if self.output is not None:
            print("Downloading '{}' at '{}'...".format(self.url, self.fpath),
                  end="")

        with urlopen(self.url) as url_handle, \
                open(self.fpath, "w+b") as file_handle:
            shutil.copyfileobj(url_handle, file_handle)

        if self.output is not None:
            print(os.linesep)


    def access(self, folder=None):
        if folder is not None:
            self.folder = folder
        if self.override or not os.path.exists(self.fpath):
            self.download()
        return self.fpath

