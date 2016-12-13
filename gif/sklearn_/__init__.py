"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""

from .gradient_boosting import GradientBoostingClassifier
from .gradient_boosting import GradientBoostingRegressor

__all__ = ["GradientBoostingClassifier",
           "GradientBoostingRegressor",]
