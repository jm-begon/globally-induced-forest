# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# License: BSD 3 clause
"""
Machine learning module for the globally inducted (decision) forest (GIF).

It is heavily based on the scikit-learn (http://scikit-learn.org) implementation
of trees.
"""

__version__ = '0.1.0'


from .forest.forest import GIFClassifier, GIFRegressor

__all__ = ["GIFClassifier", "GIFRegressor"]


