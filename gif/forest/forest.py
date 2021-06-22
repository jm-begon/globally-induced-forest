"""
This module contains the GIF models.
"""
# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# Licence: BSD 3 clause



from __future__ import division

import numbers
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError

from ._forestbuilder import TreeFactory, GIFBuilder
from . import _loss
from ..tree import _tree

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

MAX_RAND_SEED = np.iinfo(np.int32).max

__all__ = ["GIFClassifier", "GIFRegressor"]


def post_builder(cls, *args, **kwargs):
    def constructor(obj):
        dict = {k:getattr(obj, k) for k in args}
        dict.update({k:getattr(obj, v) for k, v in kwargs.items()})
        return cls(**dict)
    return constructor

LOSS_CLF = {"exponential":post_builder(_loss.ExponentialLoss),
            "severe_exponential":post_builder(_loss.SevereExponentialLoss),
            "trimmed_exponential":post_builder(_loss.TrimmedExponentialLoss,
                                               saturation='trimmed_exponential_saturation')}
LOSS_REG = {"square":post_builder(_loss.SquareLoss)}





class GIForest(BaseEstimator, metaclass=ABCMeta):
    """Base class for GIForest"""
    # Node budget

    @abstractmethod
    def __init__(self, init_pool_size,
                       dynamic_pool,
                       budget,
                       learning_rate,
                       candidate_window,
                       loss,
                       criterion,
                       splitter,
                       max_features,
                       max_depth,
                       min_samples_split,
                       min_samples_leaf,
                       min_weight_fraction_leaf,
                       max_leaf_nodes,
                       min_impurity_split,
                       class_weight,
                       presort,
                       process_pure_leaves,
                       trimmed_exponential_saturation,
                       random_state):
        self.init_pool_size = init_pool_size
        self.dynamic_pool = dynamic_pool
        self.budget = budget
        self.learning_rate = learning_rate
        self.candidate_window = 0 if candidate_window is None else candidate_window
        self.loss = loss
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
        self.process_pure_leaves = process_pure_leaves
        self.trimmed_exponential_saturation = trimmed_exponential_saturation
        self.random_state = random_state

        self.estimators_ = None
        self.history_ = None
        self.bias = None
        self.proba_transformer = None
        self.n_estimators = None
        self.actual_budget = None

        if class_weight is not None:
            raise NotImplementedError("class_weight != None unsupported")



    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression). In the regression case, use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same datasets, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        if sample_weight is not None:
            raise NotImplementedError("sample_weight != None unsupported")

        if X_idx_sorted is not None:
            raise NotImplementedError("X_idx_sorted != None unsupported")

        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            y = check_array(y, ensure_2d=False, dtype=None)
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        is_classification = isinstance(self, ClassifierMixin)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=np.int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

        else:
            self.classes_ = [None] * self.n_outputs_
            self.n_classes_ = [1] * self.n_outputs_

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not (0. < self.min_samples_split <= 1. or
                2 <= self.min_samples_split):
            raise ValueError("min_samples_split must be in at least 2"
                             " or in (0, 1], got %s" % min_samples_split)
        if not (0. < self.min_samples_leaf <= 0.5 or
                1 <= self.min_samples_leaf):
            raise ValueError("min_samples_leaf must be at least than 1 "
                             "or in (0, 0.5], got %s" % min_samples_leaf)

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        presort = self.presort
        # Allow presort to be 'auto', which means True if the datasets is dense,
        # otherwise it will be False.
        if self.presort == 'auto' and issparse(X):
            presort = False
        elif self.presort == 'auto':
            presort = True

        if presort is True and issparse(X):
            raise ValueError("Presorting is not supported for sparse "
                             "matrices.")

        # If multiple trees are built on the same datasets, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        # which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        if self.trimmed_exponential_saturation is None:
            self.trimmed_exponential_saturation = MAX_RAND_SEED

        if is_classification:
            loss = LOSS_CLF[self.loss](self)
            self.proba_transformer = loss.proba_transformer()
        else:
            loss = LOSS_REG[self.loss](self)

        min_impurity_split = self.min_impurity_split
        criterion = self.criterion
        splitter = self.splitter
        process_pure_leaves = self.process_pure_leaves

        cw_seed = random_state.randint(0, MAX_RAND_SEED)

        tree_factory = TreeFactory(min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   min_weight_leaf=min_weight_leaf,
                                   min_samples_split=min_samples_split,
                                   min_impurity_split=min_impurity_split,
                                   max_depth=max_depth,
                                   max_leaf_nodes=max_leaf_nodes,
                                   criterion_name=criterion,
                                   splitter_name=splitter,
                                   random_state=random_state,
                                   presort=presort,
                                   process_pure_leaves=process_pure_leaves)

        builder = GIFBuilder(loss, tree_factory, self.init_pool_size,
                             self.budget, self.learning_rate, self.dynamic_pool,
                             self.candidate_window, cw_seed)


        # TODO ensure mode 'c'
        trees, bias, history, n_cands = builder.build(X, y, self.n_classes_)

        # reduce history if necessary
        if history[-1] < 0 or history[-1] > len(trees):
            empty_idx = np.argmax(history == history[-1])
            history = history[:empty_idx]

        # reduce list of tree if necessay
        histogram = np.zeros(len(trees), dtype=int)
        for h in history:
            histogram[h] += 1
        trees = [t for t, h in zip(trees, histogram) if h > 0]

        # Adapt the tree indices in history
        skips = np.cumsum(histogram == 0)
        for i in range(len(history)):
            history[i] -= skips[history[i]]


        self.actual_budget = histogram.sum() + len(trees)
        self.n_estimators = len(trees)
        self.estimators_ = trees
        self.history_ = history
        self.bias = bias
        self.n_final_candidates = n_cands

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            self.bias = self.bias[0]

        return self


    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def _raw_to_predict(self, raw):
        # Bias is included !
        # Classification
        if isinstance(self, ClassifierMixin):
            n_samples = raw.shape[0]
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(raw, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(raw[:, k], axis=1),
                        axis=0)

                return predictions

        # Regression
        else:
            # Remove the `class` channel since there only 1
            if self.n_outputs_ == 1:
                return raw[:, 0]

            else:
                return raw[:, :, 0]

    def raw_predict(self, X, check_input=True):
        X = self._validate_X_predict(X, check_input)

        raw_preds = self.estimators_[0].predict(X)
        for tree_ in self.estimators_[1:]:
            raw_preds += tree_.predict(X)

        raw_preds += self.bias

        return raw_preds

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        raw_preds = self.raw_predict(X, check_input=check_input)

        return self._raw_to_predict(raw_preds)


    def apply(self, X, check_input=True):
        """Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X, check_input=check_input)

        return np.array([tree_.apply(X) for tree_ in self.estimators_]).T


    def decision_path(self, X, check_input=True):
        """Return the decision path in the forest

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.
        n_nodes_ptr : array of size (n_estimators + 1, )
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """

        X = self._validate_X_predict(X, check_input=check_input)
        indicators = [tree_.decision_path(X) for tree_ in self.estimators_]

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        raise NotImplementedError("Feature importance is not well defined in "
                                  "this context.")
        if self.estimators_ is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        importances = self.estimators_[0].compute_feature_importances().copy()
        for tree in self.estimators_[1:]:
            importances += tree.compute_feature_importances()
        return importances / float(len(self.estimators_))


    def staged_predict(self, X, copy=False):
        """
        Predict class at each stage for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input
        """
        try:
            # If self.n_classes_ is an array
            n_classes = self.n_classes_.max()
        except:
            # If self.n_classes_ is a integer
            n_classes = self.n_classes_
        raw = np.zeros((len(X), self.n_outputs_, n_classes))
        # compute parents
        parents = []
        for t_idx in range(self.n_estimators):
            tree = self.estimators_[t_idx]
            parent_array = np.ones(tree.node_count, dtype=np.int64)*(-1)
            for n_idx in range(tree.node_count):
                l_idx = tree.children_left[n_idx]
                if l_idx >= 0:
                    parent_array[l_idx] = n_idx
                r_idx = tree.children_right[n_idx]
                if r_idx >= 0:
                    parent_array[r_idx] = n_idx
            parents.append(parent_array)

        # decison path
        dpath, shifts = self.decision_path(X)
        dpath = dpath.tocsc()
        indices = dpath.indices
        indptr = dpath.indptr
        dpath = None

        # Initialisation
        raw[:] = self.bias
        n_nodes = 0

        raw_tmp = raw if self.n_outputs_ > 1 else raw.reshape(X.shape[0], self.n_classes_)
        predictions = self._raw_to_predict(raw_tmp)
        predictions = predictions.copy() if copy else predictions
        yield n_nodes, predictions

        histogram = np.zeros(self.n_estimators, dtype=np.intp)

        for t_idx in self.history_:
            tree = self.estimators_[t_idx]
            # Count node
            if histogram[t_idx] == 0:
                n_nodes += 1
            n_nodes += 1
            histogram[t_idx] += 1
            # get nodes
            n_idx = histogram[t_idx]
            ind_idx = shifts[t_idx]+n_idx
            for inst_idx in indices[indptr[ind_idx]:indptr[ind_idx+1]]:
                p_idx = parents[t_idx][n_idx]
                raw[inst_idx] -= tree.value[p_idx]
                raw[inst_idx] += tree.value[n_idx]

            raw_tmp = raw if self.n_outputs_ > 1 else raw.reshape(X.shape[0], self.n_classes_)
            predictions = self._raw_to_predict(raw_tmp)
            predictions = predictions.copy() if copy else predictions
            yield n_nodes, predictions



    def staged_n_trees(self):
        histogram = np.zeros(self.n_estimators, dtype=np.intp)
        n_trees = 0
        yield 0
        for t_idx in self.history_:
            if histogram[t_idx] == 0:
                n_trees += 1
            histogram[t_idx] += 1
            yield n_trees


class GIFClassifier(GIForest, ClassifierMixin):
    """
    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default='auto')
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    loss: string, optional (default="exponential")
        The loss to optimize while computing the nodes' weight. So far, only the
        exponential loss is supported for classification

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    #TODO attributes and stuff

    def __init__(self, init_pool_size=10,
                       dynamic_pool=False,
                       budget=10000,
                       learning_rate=1.,
                       candidate_window=None,
                       criterion="gini",
                       splitter="random",
                       max_features='auto',
                       loss="trimmed_exponential",
                       trimmed_exponential_saturation=1,
                       max_depth=None,
                       min_samples_split=2,
                       min_samples_leaf=1,
                       min_weight_fraction_leaf=0.,
                       min_impurity_split=1e-7,
                       max_leaf_nodes=None,
                       class_weight=None,
                       presort=False,
                       random_state=None):
        super(GIFClassifier, self).__init__(
            init_pool_size=init_pool_size,
            dynamic_pool=dynamic_pool,
            budget=budget,
            learning_rate=learning_rate,
            candidate_window=candidate_window,
            criterion=criterion,
            splitter=splitter,
            max_features=max_features,
            loss=loss,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_split=min_impurity_split,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            presort=presort,
            process_pure_leaves=loss=="trimmed_exponential",
            trimmed_exponential_saturation=trimmed_exponential_saturation,
            random_state=random_state)






class GIFRegressor(GIForest, RegressorMixin):
    """
    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default='auto')
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    loss: string, optional (default="square")
        The loss to optimize while computing the nodes' weight. So far, only the
        square loss is supported for regression

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, init_pool_size=10,
                       dynamic_pool=False,
                       budget=10000,
                       learning_rate=1.,
                       candidate_window=None,
                       criterion="mse",
                       splitter="random",
                       max_features='auto',
                       loss="square",
                       max_depth=None,
                       min_samples_split=2,
                       min_samples_leaf=1,
                       min_weight_fraction_leaf=0.,
                       min_impurity_split=1e-7,
                       max_leaf_nodes=None,
                       class_weight=None,
                       presort=False,
                       random_state=None):
        super(GIFRegressor, self).__init__(
            init_pool_size=init_pool_size,
            dynamic_pool=dynamic_pool,
            budget=budget,
            learning_rate=learning_rate,
            candidate_window=candidate_window,
            criterion=criterion,
            splitter=splitter,
            max_features=max_features,
            loss=loss,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_split=min_impurity_split,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            presort=presort,
            process_pure_leaves=True,
            trimmed_exponential_saturation=None,
            random_state=random_state)

