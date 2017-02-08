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
from libc.math cimport exp, log, sqrt, pow, round, fabs
from libc.float cimport DBL_MIN, DBL_MAX


cdef double __MISSING__ = -DBL_MAX / 1e10
cdef double __EPS__ = 1e-10


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


    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs,
                   SIZE_t max_n_classes):
        """
        Parameters
        ----------
        y : data from array [n_instances, n_outputs] in 'c' mode
        """
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
        self.current_weights = <double*> calloc(self.weights_size, sizeof(double))
        self.errors = <double*> calloc(self.error_size, sizeof(double))



    def __dealloc__(self):
        free(self.current_weights)
        free(self.errors)


    cdef void copy_weight(self, double* weights) nogil:
        cdef:
            SIZE_t i
            SIZE_t weights_size = self.weights_size
            double* current_weights = self.current_weights

        for i in range(weights_size):
            weights[i] = current_weights[i]

    cdef void update_errors(self, SIZE_t start, SIZE_t end, SIZE_t* indices,
                            double* deltas) nogil:
        pass

    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:
        pass

    cdef void adapt_tree_values(self, double* parent_values, double* node_values,
                               double* weights, SIZE_t* indices, SIZE_t start,
                               SIZE_t end) nogil:

        # parent_values, node_values and weights are all [n_output, n_classes]
        cdef:
            SIZE_t n_outputs = self.n_outputs
            SIZE_t n_classes = self.max_n_classes
            SIZE_t output_stride, i, j


        for i in range(n_outputs):
            output_stride = i*n_classes
            for j in range(n_classes):
                node_values[output_stride+j] = (parent_values[output_stride+j] +
                     weights[output_stride+j])




cdef class RegressionLoss(Loss):
    # Only one class
    pass

cdef class ClassificationLoss(Loss):
    def proba_transformer(self):
        pass


cdef class SquareLoss(RegressionLoss):
    # Note that the error use in this class is the residual (not the square
    # residual)

    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs,
                   SIZE_t max_n_classes):
        Loss.init(self, y, n_instances, n_outputs, max_n_classes)
        if max_n_classes != 1:
            # error
            raise ValueError("Only one 'class' in regression")

        # Compute errors for the null model
        cdef SIZE_t i
        cdef double* residuals = self.errors
        cdef SIZE_t error_size = self.error_size
        for i in range(error_size):
            residuals[i] = y[i]




    cdef void update_errors(self, SIZE_t start, SIZE_t end, SIZE_t* indices,
                            double* deltas) nogil:
        cdef double* residuals = self.errors
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t inst_stride = self.inst_stride
        cdef SIZE_t i, out, index

        # Note that out_stride = max_n_class = 1

        for i in range(start, end):
            index = indices[i] * inst_stride
            for out in range(n_outputs):
                residuals[index+out] -= deltas[out]



    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:
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

        for i in range(start, end):
            for j in range(n_outputs):
                weights[j] += residuals[indices[i]*inst_stride + j]

        for j in range(n_outputs):
            error_red += (weights[j]**2 / local_n_instances)
            weights[j] /= local_n_instances

        return error_red


cdef class ExponentialLoss(ClassificationLoss):
    # the error is the loss (i.e. exponential of margin)
    # However, the loss accounts for a compact representation: we only need
    # one slot per output !
    # This implementation assumes the same number of class per output
    cdef double* class_errors

    def __cinit__(self):
        self.class_errors = NULL

    def __dealloc__(self):
        free(self.class_errors)


    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs,
                   SIZE_t max_n_classes):
        """
        Parameters
        ----------
        y : data from array [n_instances, n_outputs] in 'c' mode
        """
        # TODO make thing clear about what array each stride relates
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
        self.current_weights = <double*> calloc(self.weights_size, sizeof(double))
        self.errors = <double*> calloc(self.error_size, sizeof(double))
        self.class_errors = <double*> calloc(self.max_n_classes, sizeof(double))

        # Compute the error for the null model
        cdef SIZE_t i, error_size = self.error_size
        cdef double* loss = self.errors
        for i in range(error_size):
            loss[i] = 1




    def proba_transformer(self):
        def TODO(raw_pred):
            return raw_pred
        return



    cdef void update_errors(self, SIZE_t start, SIZE_t end, SIZE_t* indices,
                            double* deltas) nogil:
        cdef double* loss = self.errors
        cdef double* y = self.y
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_classes = self.max_n_classes
        cdef SIZE_t n_effective_classes = self.max_n_classes
        cdef SIZE_t inst_stride = self.inst_stride
        cdef SIZE_t out_stride = self.out_stride

        cdef double inv_n_cl_1 = 1

        cdef SIZE_t i, j, out, idx, label, index

        # There is only one slot per output (i.e. max_n_class is virtually 1)
        # Both self.errors and self.y are [n_instances, n_outputs]
        # deltas is [n_outputs, n_classes]

        inv_n_cl_1 = 1./(n_classes-1)

        for out in range(n_outputs):
        #     # Adapt the number of classes
        #     n_effective_classes = n_classes
        #     for j in range(n_classes):
        #         if deltas[out*out_stride + j] <= -1e30:  # TODO value ?
        #             with gil:
        #                 print "missing class", j  # TODO remove
        #             n_effective_classes -= 1
        #
        #     inv_n_cl_1 = 1./(n_effective_classes-1)

            # Update the errors
            for i in range(start, end):
                index = indices[i]*inst_stride
                idx = index + out*out_stride
                label = <SIZE_t>(y[idx] + .5)
                loss[idx] *= exp(-inv_n_cl_1*deltas[out*n_classes + label])





    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:

        cdef:
            double geometric_mean
            double error_sum = 0
            double log_prod = 0
            double* weights = self.current_weights
            double error_reduction = 0
            SIZE_t n_outputs = self.n_outputs
            SIZE_t n_classes = self.max_n_classes
            SIZE_t n_effective_classes = self.max_n_classes
            SIZE_t inst_stride = self.inst_stride
            SIZE_t out_stride = self.out_stride
            SIZE_t cls_stride = self.cls_stride
            SIZE_t i, j, out_channel, idx
            SIZE_t label
            double* y = self.y
            double* loss = self.errors
            double* class_errors = self.class_errors


        for out_channel in range(n_outputs):
            n_effective_classes = n_classes
            error_sum = 0
            log_prod = 0

            # Reset the class error vector
            for j in range(n_classes):
                class_errors[j] = 0

            # Update the class error for the instances in [start, end]
            for i in range(start, end):
                idx = indices[i]*inst_stride
                label = <int>(y[idx+out_channel] + .5)
                class_errors[label] += loss[idx + out_channel*out_stride]


            # Compute the total error components and adapt the number of classes
            for j in range(n_classes):
                if class_errors[j] == 0.:
                    n_effective_classes -= 1
                else:
                    log_prod += log(class_errors[j])
                    error_sum  += class_errors[j]


            geometric_mean = exp(log_prod/float(n_effective_classes))

            # Adapt the weights
            for j in range(n_classes):
                if class_errors[j] == 0.:
                    weights[out_channel*n_outputs + j] = -DBL_MAX
                else:
                    weights[out_channel*n_outputs + j] = (n_effective_classes-1)*(log(class_errors[j]) - log_prod/n_effective_classes)



            # Compute the error reduction
            error_reduction += (error_sum - (n_effective_classes * geometric_mean))


        return error_reduction


    cdef void adapt_tree_values(self, double* parent_values, double* node_values,
                                double* weights, SIZE_t* indices, SIZE_t start,
                                SIZE_t end) nogil:

        # parent_values, node_values and weights are all [n_output, n_classes]
        cdef:
            SIZE_t n_outputs = self.n_outputs
            SIZE_t n_classes = self.max_n_classes
            SIZE_t output_stride, i, j


        for i in range(n_outputs):
            output_stride = i*n_classes
            for j in range(n_classes):
                if weights[output_stride+j] <= -DBL_MAX:
                    node_values[output_stride+j] = __MISSING__
                else:
                    node_values[output_stride+j] = (
                        parent_values[output_stride+j] +
                        weights[output_stride+j])
