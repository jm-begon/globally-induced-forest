
# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# Licence: BSD 3 clause


import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef class Loss:
    # The loss is responsible for evaluating the error with respect to the
    # predictions, optimizing a weight over a subset of instances and computing
    # the error reduction over that subset
    cdef:
        SIZE_t inst_stride
        SIZE_t out_stride
        SIZE_t cls_stride
        SIZE_t n_outputs
        SIZE_t n_instances
        SIZE_t max_n_classes
        SIZE_t weights_size
        SIZE_t error_size
        double* y
        double* current_weights
        double* errors


    cdef:
        void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs,
                  SIZE_t max_n_classes)

        void update_errors(self, SIZE_t start, SIZE_t end, SIZE_t* indices,
                           double* deltas) nogil

        double optimize_weight(self,
                               SIZE_t* indices,
                               SIZE_t start,
                               SIZE_t end) nogil

        void adapt_tree_values(self, double* parent_values, double* node_values,
                               double* weights, SIZE_t* indices, SIZE_t start,
                               SIZE_t end) nogil

        void copy_weight(self, double* weights) nogil


cdef class ClassificationLoss(Loss):

    cdef inline SIZE_t get_label(self, double label_as_double) nogil