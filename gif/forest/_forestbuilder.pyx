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

from ..tree._utils cimport rand_int

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

cdef class CounterVector:
    """A vector of `SIZE_t`

    Attributes
    ----------
    size : SIZE_t
        The size of the vector

    top : SIZE_t
        The actual size of the vector

    vector_ : `SIZE_t` array
        The content of the vector
    """

    def __cinit__(self, SIZE_t size):
        self.size = size
        self.vector_ = <SIZE_t*> calloc(size, sizeof(SIZE_t))
        if self.vector_ == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.vector_)


    cdef SIZE_t size(self) nogil:
        """Return the size of the vector"""
        return self.size

    cdef SIZE_t increment(self, SIZE_t index) nogil:
        """Increment and return the current count of the vector or <SIZE_t>-1
        in case of error"""
        cdef SIZE_t size = self.size
        cdef SIZE_t* vector = NULL
        cdef SIZE_t i = 0

        if index >= size:
            self.size *= 2
            vector = <SIZE_t*> realloc(self.vector_,
                                       self.size * sizeof(Candidate))
            if vector == NULL:
                # no free; __dealloc__ handles that
                return <SIZE_t>-1
            # Zeroing the new entries
            for i in range(size, size*2):
                vector[i] = 0
            self.vector_ = vector
            return self.increment(index)  # Ensure that the size is sufficient

        vector = self.vector_
        vector[index] += 1
        return vector[index]

    cdef SIZE_t get(self, SIZE_t index) nogil:
        """
        Return the element at index ``index`` or <SIZE_t>-1 in case of error
        (overflow)"""
        cdef SIZE_t size = self.size
        cdef SIZE_t* vector = self.vector_

        if index >= size:
            return <SIZE_t>-1

        return vector[index]




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

    def __cinit__(self, SIZE_t capacity, UINT32_t random_state):
        self.capacity = capacity
        self.random_state = random_state
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

    cdef void shuffle(self, SIZE_t n_first) nogil:
        """
        Shuffle the candidate list by placing `n_first` random candidate at
        the start of the list
        """
        cdef Candidate* vector = self.vector_
        cdef UINT32_t* random_state = &self.random_state

        cdef Candidate tmp
        cdef SIZE_t size = self.size()
        cdef SIZE_t i, j
        n_first = max(0, min(n_first, size))

        for i in range(n_first):
            j = rand_int(i, size, random_state)
            tmp = vector[i]
            vector[i] = vector[j]
            vector[j] = tmp





cdef class TreeFactory:
    """
    `TreeFactory`
    =============

    A factory containing all the information for building `GIFTreeBuilder`
    """

    def __init__(self, max_features, min_samples_leaf, min_weight_leaf,
                 min_samples_split, min_impurity_split, max_depth,
                 max_leaf_nodes, criterion_name, splitter_name,
                 random_state, presort, process_pure_leaves):
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion = criterion_name
        self.splitter = splitter_name
        self.random_state = random_state
        self.presort = presort
        self.min_impurity_split = min_impurity_split
        self.process_pure_leaves = process_pure_leaves


    cpdef GIFTreeBuilder build(self,
                               object X,
                               np.ndarray y,
                               np.ndarray n_classes,
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
        is_classification = n_classes[0] > 1
        n_outputs = y.shape[1]
        n_features = X.shape[1]
        max_n_classes = n_classes.max()

        random_state = check_random_state(self.random_state)
        max_features = self.max_features

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

        process_pure_leaves = self.process_pure_leaves
        tree_ = Tree(n_features, n_classes, n_outputs)
        return GIFTreeBuilder(tree_, splitter, self.min_samples_split,
                              self.min_samples_leaf, self.min_weight_leaf,
                              self.min_impurity_split, self.max_depth,
                              tree_index, n_outputs, max_n_classes,
                              process_pure_leaves)


cdef class GIFTreeBuilder:
    """A builder class which can grow the tree. Several `GIFTreeBuilder`
    interact through a commom `CandidateList`
    """

    def __cinit__(self, Tree tree, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  double min_impurity_split, SIZE_t max_depth,
                  SIZE_t tree_index, SIZE_t n_outputs, SIZE_t max_n_classes,
                  bint process_pure_leaves):
        self.tree = tree
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self.tree_index = tree_index
        self.n_outputs = n_outputs
        self.max_n_classes = max_n_classes
        self.process_pure_leaves = process_pure_leaves
        self.max_depth_seen = 0



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
            bint process_pure_leaves = self.process_pure_leaves
            SIZE_t n_outputs = self.n_outputs
            SIZE_t max_n_classes = self.max_n_classes
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
            bint is_leaf, new_node
            SIZE_t parent_idx, self_idx, output_stride, self_shift, parent_shift
            SIZE_t node_id, i, j




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



            # Compute node value.
            # Note that value is [node_count, n_outputs, max_n_classes]
            if parent == _TREE_UNDEFINED:
                for i in range(n_outputs*max_n_classes):
                    tree.value[i] = 0
            else:
                parent_idx = parent*tree.value_stride
                self_idx = node_id*tree.value_stride
                for i in range(n_outputs):
                    output_stride = i*max_n_classes
                    self_shift = self_idx+output_stride
                    parent_shift = parent_idx+output_stride
                    for j in range(max_n_classes):
                        tree.value[self_shift+j] = (
                            tree.value[parent_shift+j] +
                            weights[output_stride+j])



            if depth > max_depth_seen:
                self.max_depth_seen = depth


            new_node = False
            if not is_leaf:
                if process_pure_leaves or split.impurity_left > 0:
                    # Adding left node
                    candidates.add(start, split.pos, depth+1, node_id, True,
                                   split.impurity_left, n_constant_features,
                                   tree_index, splitter.samples)
                    new_node = True
                if process_pure_leaves or split.impurity_right > 0:
                    # Adding right node
                    candidates.add(split.pos, end, depth+1, node_id, False,
                                   split.impurity_right, n_constant_features,
                                   tree_index, splitter.samples)
                    new_node = True
                return new_node
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
        cdef SIZE_t n_node_samples = X.shape[0]
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

    candidate_window: size_t
        The number of candidates examined at each iteration. 0 will be
        interpreted as all the candidates
    """

    def __cinit__(self, Loss loss, TreeFactory tree_factory, SIZE_t n_trees,
                  SIZE_t budget, double learning_rate, bint dynamic_pool,
                  SIZE_t candidate_window, UINT32_t random_state):
        if budget < n_trees:
            budget = n_trees
        self.loss = loss
        self.tree_factory = tree_factory
        self.n_trees = n_trees
        self.budget = budget
        self.learning_rate = learning_rate
        self.dynamic_pool = dynamic_pool
        self.candidate_window = candidate_window
        self.random_state = random_state



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
            y = np.ascontiguousarray(y, dtype=DOUBLE) # Ensure C order

        return X, y



    cpdef build(self, object X, np.ndarray[DOUBLE_t, ndim=2] y,
                np.ndarray[SIZE_t, ndim=1] n_classes):
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
        cdef SIZE_t max_n_classes = n_classes.max()

        X, y = self._check_input(X, y, is_classification)


        cdef:
            # Self stuff
            Loss loss = self.loss
            TreeFactory tree_factory = self.tree_factory
            SIZE_t n_trees = self.n_trees
            SIZE_t budget = self.budget
            double learning_rate = self.learning_rate
            bint dynamic_pool = self.dynamic_pool
            SIZE_t candidate_window = self.candidate_window

            # Derived information
            SIZE_t n_instances = y.shape[0]
            SIZE_t n_outputs = y.shape[1]

            # Data structure
            CandidateList candidate_list = CandidateList(INITIAL_VECTOR_SIZE,
                                                         self.random_state)
            Candidate node
            GIFTreeBuilder tree_builder
            list tree_builders = []
            list trees = []

            # Return stuff
            np.ndarray[SIZE_t, ndim=1] history = np.ones(budget, dtype=np.intp)*(<SIZE_t>-1)
            np.ndarray[DOUBLE_t, ndim=2] intercept = np.zeros((n_outputs, max_n_classes),
                                                              dtype=DOUBLE,
                                                              order='C')
            double* intercept_ptr = <double*>intercept.data


            # Numpy's arrays and pointers
            np.ndarray[DOUBLE_t, ndim=1] weights = np.zeros(n_outputs*max_n_classes,
                                                            dtype=DOUBLE)
            double* weights_ptr = <double*>weights.data
            np.ndarray[SIZE_t, ndim=1] inst_id = np.arange(n_instances,
                                                           dtype=np.intp)
            SIZE_t* inst_id_ptr = <SIZE_t*>inst_id.data

            CounterVector histogram = CounterVector(n_trees)


            # Other variables
            SIZE_t i, b = 0, best_cand_idx, cand_idx, size, n_nodes = 0
            SIZE_t n_candidates = 0



        # Initializing the loss
        loss.init(<double*>y.data, n_instances, n_outputs, max_n_classes)
        loss.optimize_weight(inst_id_ptr, 0, n_instances)
        loss.copy_weight(intercept_ptr)
        loss.update_errors(0, n_instances, inst_id_ptr, intercept_ptr)
        # Make some space
        inst_id_ptr = NULL
        del inst_id

        # Building the stumps
        for i in range(n_trees):
            tree_builder = tree_factory.build(X, y, n_classes, i)
            tree_builder.make_stump(X, y, candidate_list) # TODO  -- nogil ?
            tree_builders.append(tree_builder)

        # Main loop
        while n_nodes < budget:
            error_reduction = float("-inf")
            best_cand_idx = 0
            with nogil:
                size = candidate_list.size()
                if size == 0:
                    with gil:
                        print "No more candidate"
                    break
                # Shuffle if necessary
                if candidate_window > 0 and candidate_window < size:
                    n_candidates = candidate_window
                    candidate_list.shuffle(n_candidates)
                else:
                    n_candidates = size
                # Find the best candidate
                for cand_idx in range(n_candidates):
                    candidate_list.peek(cand_idx, &node)
                    cand_error_red = loss.optimize_weight(node.indices,
                                                          node.start,
                                                          node.end)
                    if cand_error_red > error_reduction:
                        error_reduction = cand_error_red
                        loss.copy_weight(weights_ptr)
                        best_cand_idx = cand_idx


                if error_reduction <= 0:
                    # Can we ensure this for all losses ?
                    with gil:
                        print "error_reduction <= 0"
                    break

                # Extract candidate
                candidate_list.pop(best_cand_idx, &node)
                t_idx = node.tree_index

                # Adapt error vector
                for i in range(n_outputs*max_n_classes):
                    weights_ptr[i] *= learning_rate


            # Add candidate's children to the list
            tree_builder = tree_builders[t_idx]

            tree_builder.add_and_develop(&node,
                                         weights_ptr,
                                         candidate_list)

            # Compute the actual number of trees and nodes
            with nogil:
                history[b] = t_idx

                new_count = histogram.increment(t_idx)
                if new_count == 1:  # A new root has been added
                    n_nodes += 1
                    if dynamic_pool:
                        with gil:
                            tree_builder = tree_factory.build(X, y, n_classes,
                                                              n_trees)
                            tree_builder.make_stump(X, y, candidate_list)
                            tree_builders.append(tree_builder)
                        n_trees += 1
                n_nodes += 1
                b += 1


            # Update the loss
            loss.update_errors(node.start, node.end, node.indices, weights_ptr)


        # Finalize
        for i in range(n_trees):
            tree_builder = tree_builders[i]
            tree_builder.finalize()
            trees.append(tree_builder.tree)

        return trees, intercept, history
