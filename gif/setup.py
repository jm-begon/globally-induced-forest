import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("gif", parent_package, top_path)

    config.add_subpackage("tree")
    config.add_subpackage("forest")
    config.add_subpackage("sklearn_")
    config.add_subpackage("datasets")

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
