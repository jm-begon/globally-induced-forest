# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# Licence: BSD 3 clause

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


# =============================================================================
# Candidate data structure
# =============================================================================

# Information for Candidate
cdef struct Candidate:
    SIZE_t start                        # Start (inclusive) index in `samples`
    SIZE_t end                          # End (exclusive) index in `samples`
    SIZE_t depth                        # Depth of that node
    SIZE_t parent                       # Index of the parent node
    bint is_left                        # Whether it is a left node
    double impurity                     # Impurity measure
    SIZE_t n_constant_features          # Number of constant features
    SIZE_t tree_index                   # Index of the tree
    SIZE_t* samples                     # Array of indices to y of the node's
                                        # training instances


# Candidate vector
cdef class CandidateList:
    cdef SIZE_t capacity                # Capacity of the vector
    cdef SIZE_t top                     # Actual size of the vector
    cdef Candidate* vector_             # Array containing the entries


    cdef int add(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features,
                  SIZE_t tree_index, SIZE_t* samples) nogil

    cdef int pop(self, SIZE_t index, Candidate* res) nogil

    cdef int peek(self, SIZE_t index, Candidate* res) nogil


# =============================================================================
# GIFTreeBuilder
# =============================================================================

cdef class GIFTreeBuilder:

    cdef Tree tree                      # The tree to develop
    cdef Splitter splitter              # The splitter to build the tree
    cdef SIZE_t min_samples_split
    cdef SIZE_t min_samples_leaf
    cdef double min_weight_leaf
    cdef SIZE_t max_depth
    cdef SIZE_t tree_index
    cdef SIZE_t n_outputs
    cdef SIZE_t max_depth_seen

    cdef bint develop_node(self, SIZE_t start, SIZE_t end, SIZE_t depth,
                           SIZE_t parent, bint is_left,
                           double impurity,
                           SIZE_t n_constant_features,
                           SIZE_t, tree_index,
                           CandidateList candidates,
                           double* weights) nogil

    cdef inline bint add_and_develop(self, Candidate* node,
                                     CandidateList candidate_list,
                                     double* weights) nogil

    cdef bint make_stump(self, object X, np.ndarray y,
                         CandidateList candidate_list) nogil

    cdef int finalize(self)


# =============================================================================
# GIFBuilder
# =============================================================================
cdef class GIFBuilder:

    cdef Loss loss
    cdef TreeFactory tree_factory
    cdef SIZE_t n_trees
    cdef SIZE_t budget
    cdef double learning_rate

