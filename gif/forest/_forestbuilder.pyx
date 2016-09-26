# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# Licence: BSD 3 clause

import numpy as np
cimport numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.stdlib cimport malloc
from libc.stdlib cimport calloc
from libc.string cimport memcpy
from libc.string cimport memset

from ..tree._criterion cimport Criterion
from ..tree._splitter cimport SplitRecord
from ._loss cimport ClassificationLoss

import numbers
import six
from scipy.sparse import issparse
from sklearn.utils import check_random_state

from ..tree import _tree, _splitter, _criterion

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
CRITERIA_REG = {"mse": _criterion.MSE, "friedman_mse": _criterion.FriedmanMSE}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter,
                   "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {"best": _splitter.BestSparseSplitter,
                    "random": _splitter.RandomSparseSplitter}

cdef double INFINITY = np.inf

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_VECTOR_SIZE = 10



# =============================================================================
# Classes and Algorithms
# =============================================================================

cdef class CandidateList:
    """A vector of `Candidate

    Attributes
    ----------
    capacity : SIZE_t
        The capacity of the vector

    top : SIZE_t
        The actual size of the vector

    vector_ : `Candidate` pointer
        The content of the vector
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.vector_ = <Candidate*> malloc(capacity * sizeof(Candidate))
        if self.vector_ == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.vector_)


    cdef SIZE_t size(self) nogil:
        """Return the size of the vector"""
        return self.top

    cdef int add(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features,
                  SIZE_t tree_index, SIZE_t* samples) nogil:
        """add a new element to the vector.

        Returns 0 if successful; -1 on out of memory error.
        """
        cdef SIZE_t top = self.top
        cdef Candidate* vector = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            vector = <Candidate*> realloc(self.vector_,
                                         self.capacity * sizeof(Candidate))
            if vector == NULL:
                # no free; __dealloc__ handles that
                return -1
            self.vector_ = vector

        vector = self.vector_
        vector[top].start = start
        vector[top].end = end
        vector[top].depth = depth
        vector[top].parent = parent
        vector[top].is_left = is_left
        vector[top].impurity = impurity
        vector[top].n_constant_features = n_constant_features
        vector[top].tree_index = tree_index
        vector[top].indices = samples

        # Increment vector pointer
        self.top = top + 1
        return 0

    cdef int pop(self, SIZE_t index, Candidate* res) nogil:
        """Remove the element from the vector and index ``index`` and copy it
        to ``res``.

        Returns 0 if pop was successful (and ``res`` is set); -1
        otherwise.
        """
        cdef SIZE_t top = self.top
        cdef Candidate* vector = self.vector_
        cdef Candidate tmp

        if top <= 0 or index >= top:
            return -1

        res[0] = vector[index]
        vector[index] = vector[top-1]

        self.top = top - 1

        return 0

    cdef int peek(self, SIZE_t index, Candidate* res) nogil:
        """
        Copy the element at index ``index`` but do not remove it.

        Returns 0 if pop was successful (and ``res`` is set); -1
        otherwise.
        """
        cdef SIZE_t top = self.top
        cdef Candidate* vector = self.vector_

        if top <= 0 or index >= top:
            return -1

        res[0] = vector[index]
        return 0


cdef class TreeFactory:
    """
    `TreeFactory`
    =============

    A factory containing all the information for building `GIFTreeBuilder`
    """

    def __init__(self, max_features, min_samples_leaf, min_weight_leaf,
                 min_samples_split, min_impurity_split, max_depth,
                 max_leaf_nodes, criterion_name, random_state, presort):
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion_name = criterion_name
        self.random_state = random_state
        self.presort = presort
        self.min_impurity_split = min_impurity_split


    cpdef GIFTreeBuilder build(self,
                               object X,
                               np.ndarray y,
                               SIZE_t n_classes,
                               SIZE_t tree_index):
        """
        Parameters
        ----------
        tree_index : unsigned int
            The identifier of the tree

        Return
        ------
        gif_tree_builder : `GIFTreeBuilder`
            A tree builder
        """
        is_classification = n_classes > 1
        n_outputs = y.shape[1]
        n_features = X.shape[1]

        random_state = check_random_state(self.random_state)
        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(n_features)))
                else:
                    max_features = n_features
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = n_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * n_features))
            else:
                max_features = 0

        if is_classification:
            criterion = CRITERIA_CLF[self.criterion](n_outputs, n_classes)
        else:
            criterion = CRITERIA_REG[self.criterion](n_outputs)
        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS
        splitter = SPLITTERS[self.splitter](criterion,
                                            max_features,
                                            self.min_samples_leaf,
                                            self.min_weight_leaf,
                                            random_state,
                                            self.presort)

        tree_ = Tree(n_features, n_classes, self.n_outputs)
        return GIFTreeBuilder(tree_, splitter, self.min_samples_split,
                              self.min_samples_leaf, self.min_weight_leaf,
                              self.min_impurity_split, self.max_depth,
                              tree_index)


cdef class GIFTreeBuilder:
    """A builder class which can grow the tree. Several `GIFTreeBuilder`
    interact through a commom `CandidateList`
    """

    def __cinit__(self, Tree tree, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  double min_impurity_split, SIZE_t max_depth,
                  SIZE_t tree_index, SIZE_t n_outputs):
        self.tree = tree
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self.tree_index = tree_index
        self.n_outputs = n_outputs
        self.max_depth_seen = -1


    cdef inline bint develop_node(self, SIZE_t start, SIZE_t end, SIZE_t depth,
                                  SIZE_t parent, bint is_left,
                                  double impurity,
                                  SIZE_t n_constant_features,
                                  SIZE_t tree_index,
                                  CandidateList candidates,
                                  double* weights):
        """Develop the node corresponding to the given parameters
        and place its children in the `CandidateList` if any.

        Return 1 if the node has children. 0 Otherwise
        """
        cdef:
            # Self stuff
            Splitter splitter = self.splitter
            Tree tree = self.tree
            SIZE_t n_outputs = self.n_outputs
            SIZE_t max_depth = self.max_depth
            SIZE_t min_samples_leaf = self.min_samples_leaf
            SIZE_t min_samples_split = self.min_samples_split
            SIZE_t max_depth_seen = self.max_depth_seen
            double min_weight_leaf = self.min_weight_leaf
            double min_impurity_split = self.min_impurity_split

            # Splitter stuff
            double weighted_n_samples = splitter.weighted_n_samples
            SIZE_t n_node_samples = splitter.n_samples

            # Other stuff
            SplitRecord split
            double threshold
            double weighted_n_node_samples
            bint is_leaf
            SIZE_t i, parent_idx, self_idx, node_id


        with nogil:
            n_node_samples = end - start
            splitter.node_reset(start, end, &weighted_n_node_samples)

            is_leaf = ((depth >= max_depth) or
                       (n_node_samples < min_samples_split) or
                       (n_node_samples < 2 * min_samples_leaf) or
                       (weighted_n_node_samples < min_weight_leaf))

            if parent == _TREE_UNDEFINED:  # Root node
                impurity = splitter.node_impurity()

            is_leaf = is_leaf or (impurity <= min_impurity_split)

            if not is_leaf:
                splitter.node_split(impurity, &split, &n_constant_features)
                is_leaf = is_leaf or (split.pos >= end)

            node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                     split.threshold, impurity, n_node_samples,
                                     weighted_n_node_samples)



            # Compute node value
            if parent == _TREE_UNDEFINED:
                for i in range(n_outputs):
                    tree.value[i] = 0
            else:
                parent_idx = parent*tree.value_stride
                self_idx = node_id*tree.value_stride
                for i in range(n_outputs):
                    tree.value[self_idx+i] = tree.value[parent_idx+i] + weights[i]



            if depth > max_depth_seen:
                self.max_depth_seen = depth


            if not is_leaf:
                # Adding left node
                candidates.add(start, split.pos, depth+1, node_id, True,
                               split.impurity_left, n_constant_features,
                               tree_index, splitter.samples)
                # Adding right node
                candidates.add(split.pos, end, depth+1, node_id, False,
                               split.impurity_right, n_constant_features,
                               tree_index, splitter.samples)
                return True
        return False




    cdef bint add_and_develop(self,
                              Candidate* node,
                              double* weights,
                              CandidateList candidate_list):
        return self.develop_node(node.start,
                                 node.end,
                                 node.depth,
                                 node.parent,
                                 node.is_left,
                                 node.impurity,
                                 node.n_constant_features,
                                 node.tree_index,
                                 candidate_list,
                                 weights)

    cdef bint make_stump(self, object X, np.ndarray y,
                         CandidateList candidate_list):
        """Create a stump for the (X, y) learning set and place
        their children (if any) in the `CandidateList`

        Return 1 if the stump has children. 0 otherwise
        """
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef SIZE_t t_idx = self.tree_index
        cdef bint has_children = False
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef Tree tree = self.tree


        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        self.tree._resize(init_capacity)

        splitter.init(X, y, NULL, None)

        return self.develop_node(0, n_node_samples, 0, _TREE_UNDEFINED,
                                 0, INFINITY, 0, t_idx, candidate_list, NULL)

    cdef int finalize(self):
        """Finalize the tree

        Return <0 in case of error. >=0 otherwise"""
        cdef int rc
        cdef Tree tree = self.tree
        cdef SIZE_t max_depth_seen = self.max_depth_seen

        rc = tree._resize_c(tree.node_count)
        if rc >= 0:
            rc = tree.max_depth = max_depth_seen
        return rc





cdef class GIFBuilder:
    """Forest builder. It coordinates the tree's individual `GIFTreeBuilder`
    through the `CandidateList` with the help of a `Loss`

    Attributes
    ----------
    loss : `Loss`
        The loss responsible for choosing the node and fitting their weight

    tree_factory : `TreeFactory`
        The factory which produces `GIFTreeBuilder`

    n_trees : size_t
        The number of trees

    budget : size_t
        The maximum number of nodes

    learning_rate : double
        The learning_rate used when fitting the nodes
    """

    def __cinit__(self, Loss loss, TreeFactory tree_factory, SIZE_t n_trees,
                  SIZE_t budget, double learning_rate):
        self.loss = loss
        self.tree_factory = tree_factory
        self.n_trees = n_trees
        self.budget = budget
        self.learning_rate = learning_rate


    cdef inline _check_input(self, object X, np.ndarray y,
                             bint is_classification):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.ndim != 2:
            raise ValueError("y must be 2D array [n_instances, n_outputs]")

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if is_classification:
            y_tmp = np.asarray(np.round(y), dtype=int)
            max_n_classes = 0
            for k in range(y.shape[1]):
                classes_k, _ = np.unique(y_tmp[:, k])
                max_n_classes = max(max_n_classes, len(classes_k))
        else:
            max_n_classes = 1

        return X, y, max_n_classes



    cpdef build(self, object X, np.ndarray[DOUBLE_t, ndim=2, mode="c"] y):
        """Builds the forest fot the given (X, y) learning set.

        Parameters
        ----------
        X : array of double [n_instances, n_features]
            The learning matrix
        y : array of double [n_instances, n_outputs]
            The target values

        Return
        ------
        trees : list of `Tree`
            The trees build by the algorithm
        intercept : array [n_outputs, n_classes]
            The intercept for all trees
        history : array [self.budget]
            The tree index history
        """

        cdef bint is_classification = isinstance(self.loss, ClassificationLoss)
        cdef SIZE_t max_n_classes

        X, y, max_n_classes = self._check_input(X, y, is_classification)


        cdef:
            # Self stuff
            Loss loss = self.loss
            TreeFactory tree_factory = self.tree_factory
            SIZE_t n_trees = self.n_trees
            SIZE_t budget = self.budget
            double learning_rate = self.learning_rate

            # Derived information
            SIZE_t n_instances = y.shape[0]
            SIZE_t n_outputs = y.shape[1]

            # Data structure
            CandidateList candidate_list = CandidateList(INITIAL_VECTOR_SIZE)
            Candidate node
            GIFTreeBuilder tree_builder
            list tree_builders = []
            list trees = []

            # Return stuff
            np.ndarray[SIZE_t, ndim=1] history = np.zeros(budget, dtype=int)
            np.ndarray[DOUBLE_t, ndim=2] intercept = np.zeros((n_outputs, max_n_classes),
                                                              dtype=DOUBLE,
                                                              mode='c')
            double* intercept_ptr = <double*>intercept.data

            # Other variables
            SIZE_t i,b, best_cand_idx, cand_idx, size

            # C arrays
            np.ndarray[DOUBLE_t, ndim=1] weights = np.zeros(n_outputs*max_n_classes,
                                                            dtype=DOUBLE)
            double* weights_ptr = <double*>weights.data


        loss.init(<double*>y.data, n_instances, n_outputs, max_n_classes)
        loss.optimize_weight(NULL, 0, n_instances)
        loss.copy_weight(intercept_ptr)
        for i in range(n_trees):
            tree_builder = tree_factory.build(X, y, max_n_classes, i)
            left, right = tree_builder.make_stump(X, y, candidate_list) # TODO  -- nogil

        for b in range(budget-n_trees):
            error_reduction = float("-inf")
            best_cand_idx = 0
            with nogil:
                size = candidate_list.size()
                if size == 0:
                    break
                for cand_idx in range(size):
                    candidate_list.peek(cand_idx, &node)
                    cand_error_red = loss.optimize_weight(node.indices,
                                                          node.start,
                                                          node.end)
                    if cand_error_red > error_reduction:
                        error_reduction = cand_error_red
                        loss.copy_weight(weights_ptr)
                        best_cand_idx = cand_idx

                candidate_list.pop(cand_idx, &node)
                t_idx = node.tree_index

                for i in range(n_outputs*max_n_classes):
                    weights_ptr[i] *= learning_rate

            tree_builder = tree_builders[t_idx]
            history[b] = t_idx

            tree_builder.add_and_develop(&node,
                                         weights_ptr,
                                         candidate_list)

            with nogil:

                for i in range(node.start, node.end):
                    loss.update_errors(node.indices[i], weights_ptr)


        for i in range(n_trees):
            tree_builder = tree_builders[i]
            tree_builder.finalize()
            trees.append(tree_builder.tree)

        # TODO cut through history if necessary
        #      cut through number of trees if necessary
        return trees, intercept, history




