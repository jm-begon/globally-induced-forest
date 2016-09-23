# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# Licence: BSD 3 clause

import numpy as np
cimport numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.stdlib cimport calloc
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport exp, log, sqrt, pow, round
from libc.float cimport DBL_MIN


cdef class Loss:

    def __cinit__(self):
        self.inst_stride = 0
        self.out_stride = 0
        self.cls_stride = 0
        self.n_outputs = 0
        self.n_instances = 0
        self.max_n_classes = 0
        self.y = NULL
        self.current_weights = NULL
        self.errors = NULL


    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs
                   SIZE_t max_n_classes):
        """
        Parameters
        ----------
        y : data from array [n_instances, n_outputs] in 'c' mode
        """
        self.empty()
        self.inst_stride = n_outputs*max_n_classes
        self.out_stride = max_n_classes
        self.cls_stride = 1
        self.n_outputs = n_outputs
        self.n_instances = n_instances
        self.max_n_classes = max_n_classes
        self.weights_size = n_outputs*max_n_classes
        self.error_size = n_instances*n_outputs*max_n_classes
        self.y = y
        free(self.current_weights)
        free(self.errors)
        self.current_weights = <double*> calloc(weights_size, sizeof(double))
        self.errors = <double*> calloc(error_size, sizeof(double))



    def __dealloc__(self):
        free(self.current_weights)
        free(self.errors)


    cdef void copy_weight(double* weights) nogil:
        cdef SIZE_t i
        for i in range(self.weights_size):
            weights[i] = self.current_weights[i]




cdef class RegressionLoss:
    # Only one class
    pass

cdef class ClassificationLoss:
    cpdef proba_transformer(self):
        pass


cdef class SquareLoss(RegressionLoss):
    # Note that the error use in this class is the residual (not the square
    # residual)

    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs
                   SIZE_t max_n_classes):
        Loss.init(self, y, n_instances, n_outputs, max_n_classes)
        if max_n_classes != 1:
            # error
            raise ValueError("Only one 'class' in regression")

        # Compute errror for the null model
        cdef SIZE_t i
        cdef double* residuals = self.errors
        cdef SIZE_t error_size = self.error_size
        for i in range(error_size):
            residuals[i] = -y[i]

    cdef void update_errors(SIZE_t index, double* deltas) nogil:
        cdef double* residuals = self.errors
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i
        index *= self.inst_stride

        # Note that out_stride = max_n_class = 1

        for i in range(n_outputs):
            residuals[index+i] -= deltas[i]



    cdef double optimize_weight(SIZE_t* indices, SIZE_t start, SIZE_t, end) nogil:
        cdef:
            double* residuals = self.errors
            double* weights = self.current_weights
            double error_red = 0
            double local_n_instances = end-start
            SIZE_t inst_stride = self.inst_stride
            SIZE_t n_outputs = self.n_outputs

            SIZE_t i, j

        for j in range(n_outputs):
            weights[j] = 0

        if indices == NULL: # At the start
            for i in range(start, end):
                for j in range(n_outputs):
                    weights[j] += residuals[i*inst_stride + j]
        else:
            for i in range(start, end):
                for j in range(n_outputs):
                    weights[j] += residuals[indices[i]*inst_stride + j]

        for j in range(n_outputs):
            error_red =+ (weights[j]**2 / local_n_instances)
            weights[j] /= local_n_instances

        return error_red


cdef class ExponentialLoss(ClassificationLoss):
    # the error is the loss (i.e. exponential of margin)
    # However, the loss accounts for a compact representation: we only need
    # one slot per output !
    # Assume same number of class per output
    cdef double* class_errors

    def __cinit__(self):
        class_errors = NULL

    def __dealloc__(self):
        free(class_errors)


    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs
                   SIZE_t max_n_classes):
        """
        Parameters
        ----------
        y : data from array [n_instances, n_outputs] in 'c' mode
        """
        # We only need
        self.empty()
        self.inst_stride = n_outputs
        self.out_stride = 1
        self.cls_stride = 1
        self.n_outputs = n_outputs
        self.n_instances = n_instances
        self.max_n_classes = max_n_classes
        self.weights_size = n_outputs*max_n_classes
        self.error_size = n_instances*n_outputs
        self.y = y
        free(self.current_weights)
        free(self.errors)
        free(self.class_errors)
        self.current_weights = <double*> calloc(weights_size, sizeof(double))
        self.errors = <double*> calloc(error_size, sizeof(double))
        self.class_errors = <double*> calloc(max_n_classes, sizeof(double))

        # Compute the error for the null model
        cdef SIZE_t i
        cdef double* loss = self.errors
        for i in range(error_size):
            loss[i] = 1




    cpdef proba_transformer(self):
        def TODO(raw_pred):
            return raw_pred
        return



    cdef void inline update_errors(SIZE_t index, double* deltas) nogil:
        cdef double* loss = self.errors
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_classes = self.max_n_classes
        cdef SIZE_t n_cl_1 = n_classes-1
        cdef SIZE_t i,j
        cdef SIZE_t inst_stride = self.inst_stride
        cdef SIZE_t out_stride = self.out_stride
        cdef SIZE_t cls_stride = self.cls_stride
        cdef SIZE_t idx


        for i in range(n_outputs):
            index *= inst_stride
            for j in range(n_classes):
                idx = index+i*out_stride+j*cls_stride
                loss[idx] *= exp(-n_cl_1*deltas[i*out_stride + j*cls_stride])



    cdef double optimize_weight(SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t, end) nogil):

        cdef:
            double error_prod = 1
            double error_sum = 0
            double log_prod = 0
            double* weights = self.current_weights
            double* errors = self.errors
            double missing = DBL_MIN
            double error_reduction = 0
            cdef SIZE_t n_outputs = self.n_outputs
            cdef SIZE_t n_classes = self.max_n_classes
            cdef SIZE_t inst_stride = self.inst_stride
            cdef SIZE_t out_stride = self.out_stride
            cdef SIZE_t cls_stride = self.cls_stride
            SIZE_t y_stride = self.y_stride
            SIZE_t n_outputs = self.n_outputs
            SIZE_t i, j, opt, idx
            SIZE_t label
            double* y = self.y
            double* loss = self.errror


        for out in range(n_outputs):

            error_prod = 1
            error_sum = 0
            log_prod = 0

            for j in range(n_classes):
                class_errors[j] = 0

            if indices == NULL:
                # Root node: all error equal to one
                for i in range(start, end):
                    label = <int>round(y[i])
                    class_errors[label] += 1
            else:
                for i in range(start, end):
                    idx = indices[i]
                    label = <int>round(y[idx])
                    class_errors[label] += loss[idx*inst_stride + out*out_stride]


            for j in range(n_classes):
                if class_errors[j] == 0:
                    class_errors[j] = missing
                error_prod *= class_errors[j]
                error_sum  += class_errors[j]

            log_prod = log(error_prod)
            for j in range(n_classes):
                weights[out*n_outputs+ j] = (n_classes-1)*(log(class_errors[j]) - log_prod)/n_classes


            error_reduction += error_sum - (n_classes)*pow(error_prod, 1./n_classes)
        return error_reduction











