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
            SIZE_t out_channel, i, j


        for i in range(n_outputs):
            out_channel = i*n_classes
            for j in range(n_classes):
                node_values[out_channel+j] = (parent_values[out_channel+j] +
                     weights[out_channel+j])




cdef class RegressionLoss(Loss):
    # Only one class
    pass

cdef class ClassificationLoss(Loss):
    def proba_transformer(self):
        pass

    cdef inline SIZE_t get_label(self, double label_as_double) nogil:
        return <SIZE_t>(label_as_double + .5)


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
    cdef double missing

    def __cinit__(self):
        self.class_errors = NULL
        self.missing = -1

    def __dealloc__(self):
        free(self.class_errors)
        # current_weights and errors deallocated by the parent class


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

        cdef double inv_n_cl_1 = 1./(n_classes-1)

        cdef SIZE_t i, j, out, idx, label, index

        # There is only one slot per output (i.e. max_n_class is virtually 1)
        # Both self.errors and self.y are [n_instances, n_outputs]
        # deltas is [n_outputs, n_classes]

        for out in range(n_outputs):

            # Update the errors
            for i in range(start, end):
                index = indices[i]*inst_stride
                idx = index + out*out_stride
                label = <SIZE_t>(y[idx] + .5)
                loss[idx] *= exp(-inv_n_cl_1*deltas[out*n_classes + label])



    cdef inline void _compute_class_error(self,
                                          SIZE_t* indices,
                                          SIZE_t start,
                                          SIZE_t end,
                                          SIZE_t out_channel) nogil:

        cdef:
            double* y = self.y
            double* loss = self.errors
            double* class_errors = self.class_errors
            SIZE_t n_classes = self.max_n_classes
            SIZE_t inst_stride = self.inst_stride
            SIZE_t out_stride = self.out_stride
            SIZE_t i, idx, label


        # Reset the class error vector
        for i in range(n_classes):
            class_errors[i] = 0


        # Update the class error for the instances in [start, end]
        for i in range(start, end):
            idx = indices[i]*inst_stride
            label = <int>(y[idx+out_channel] + .5)
            class_errors[label] += loss[idx + out_channel*out_stride]



    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:

        cdef:
            double* class_errors = self.class_errors
            double geometric_mean
            double error_sum = 0
            double log_prod = 0
            double* weights = self.current_weights
            double error_reduction = 0
            double missing = self.missing
            SIZE_t n_outputs = self.n_outputs
            SIZE_t n_classes = self.max_n_classes
            SIZE_t n_effective_classes = self.max_n_classes
            SIZE_t inst_stride = self.inst_stride
            SIZE_t out_stride = self.out_stride
            SIZE_t k, out_channel

        # weights are [n_output, n_classes]


        for out_channel in range(n_outputs):
            missing = self.missing
            n_effective_classes = n_classes
            error_sum = 0.
            log_prod = 0.

            # Compute the class error
            self._compute_class_error(indices, start, end, out_channel)


            # Compute the total error components and adapt the number of classes
            for k in range(n_classes):
                if class_errors[k] == 0.:
                    n_effective_classes -= 1
                else:
                    log_prod += log(class_errors[k])
                    error_sum  += class_errors[k]


            geometric_mean = exp(log_prod/float(n_effective_classes))

            # Adapt the weights for the classes represented
            for k in range(n_classes):
                if class_errors[k] > 0.:
                    weights[out_channel*n_classes + k] = (n_effective_classes-1)*(log(class_errors[k]) - log_prod/n_effective_classes)
                    if weights[out_channel*n_classes + k] < missing:
                        missing = weights[out_channel*n_classes + k]

            # Adapt the weights for the missing classes
            for k in range(n_classes):
                if class_errors[k] == 0.:
                    weights[out_channel*n_classes + k] = missing


            # Compute the error reduction
            error_reduction += (error_sum - (n_effective_classes * geometric_mean))


        return error_reduction

cdef class SevereExponentialLoss(ExponentialLoss):

    def __cinit__(self):
        self.missing = -DBL_MAX



    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:

        cdef:
            double* class_errors = self.class_errors
            double geometric_mean
            double error_sum = 0
            double log_prod = 0
            double* weights = self.current_weights
            double error_reduction = 0
            double missing = self.missing
            SIZE_t n_outputs = self.n_outputs
            SIZE_t n_classes = self.max_n_classes
            SIZE_t n_effective_classes = self.max_n_classes
            SIZE_t inst_stride = self.inst_stride
            SIZE_t out_stride = self.out_stride
            SIZE_t k, out_channel



        for out_channel in range(n_outputs):
            n_effective_classes = n_classes
            error_sum = 0.
            log_prod = 0.

            # Compute the class error
            self._compute_class_error(indices, start, end, out_channel)


            # Compute the total error components and adapt the number of classes
            for k in range(n_classes):
                if class_errors[k] == 0.:
                    n_effective_classes -= 1
                else:
                    log_prod += log(class_errors[k])
                    error_sum  += class_errors[k]


            geometric_mean = exp(log_prod/float(n_effective_classes))

            # Adapt the weights for the classes represented
            for k in range(n_classes):
                if class_errors[k] == 0.:
                    weights[out_channel*n_classes + k] = missing
                else:
                    weights[out_channel*n_classes + k] = (n_effective_classes-1)*(log(class_errors[k]) - log_prod/n_effective_classes)

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
            double missing = self.missing
            SIZE_t out_channel, i, j


        for i in range(n_outputs):
            out_channel = i*n_classes
            for j in range(n_classes):
                if weights[out_channel+j] <= missing:
                    node_values[out_channel+j] = __MISSING__
                else:
                    node_values[out_channel+j] = (
                        parent_values[out_channel+j] +
                        weights[out_channel+j])



cdef class TrimmedExponentialLoss(ExponentialLoss):
    cdef double saturation
    cdef double threshold

    def __cinit__(self, double saturation):
        self.saturation = saturation
        self.threshold =  exp(saturation)


    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:

        cdef:
            double saturation = self.saturation
            double threshold = self.threshold
            double* weights = self.current_weights
            double* class_errors = self.class_errors
            SIZE_t n_outputs = self.n_outputs
            SIZE_t n_classes = self.max_n_classes
            SIZE_t out_stride = self.out_stride

            double error_gap = 0
            double log_ratio, weight, sum_log_err
            double inv_class = 1./n_classes, inv_class_1 = 1./(n_classes-1)
            SIZE_t k, l, out_channel
            SIZE_t label



        for out_channel in range(n_outputs):
            # Compute the class error
            self._compute_class_error(indices, start, end, out_channel)

            # Reset the weights for that output
            for k in range(n_classes):
                weights[out_channel*n_classes + k] = 0


            # Compute the weights: k relates to numerator, l to denominator
            for k in range(n_classes):
                sum_log_err = 0
                for l in range(k+1, n_classes):
                    # Trim
                    if class_errors[k] == 0:
                        if class_errors[l] == 0:
                            # log(0/0)
                            log_ratio = 0
                        else:
                            # log(0/y_l)
                            log_ratio = -saturation
                    elif class_errors[l] == 0:
                        # log(x_k/0)
                        log_ratio = saturation
                    elif class_errors[k] > class_errors[l]*threshold:
                        # log(x_k/y_l) > saturation
                        log_ratio = saturation
                    elif class_errors[l] > class_errors[k]*threshold:
                        # log(x_k/y_l) < saturation
                        log_ratio = -saturation
                    else:
                        # No trimming
                        log_ratio = log(class_errors[k]) - log(class_errors[l])
                    weights[out_channel*n_classes + k] += log_ratio
                    weights[out_channel*n_classes + l] -= log_ratio
                # Normalize weight (tmp is just a shortcut)
                weight = weights[out_channel*n_classes + k]
                weight = weight - inv_class*weight
                weights[out_channel*n_classes + k] = weight

                # Adapt error gap
                tmp = exp(-inv_class_1*weight)
                error_gap += (class_errors[k]*(1-tmp))


        return error_gap





cdef class RectifiedExponentialLoss(ClassificationLoss):
    # Instead of the error, we will keep the current prediction for numerical
    # reasons


    cdef double* buffer             # Multipurpose [n_classes, n_ouput]
    cdef double* predictions        # Alias for errors
    cdef SIZE_t predictions_size    # Alais for error_size

    def __cinit__(self):
        self.predictions_size = self.error_size
        self.predictions = self.errors
        self.buffer = NULL

    def __dealloc__(self):
        # current_weights and errors are deallocated by the parent class
        free(self.buffer)


    cdef void init(self, DOUBLE_t* y, SIZE_t n_instances, SIZE_t n_outputs,
                   SIZE_t max_n_classes):
        """
        Parameters
        ----------
        y : data from array [n_instances, n_outputs] in 'c' mode
        """
        # errors (and its alias predictions) is [n_instances, n_output, n_classes]
        # current_weights is [n_output, n_classes]
        # buffer is [n_output, n_classes]
        self.cls_stride = 1
        self.out_stride = n_outputs
        self.inst_stride = n_outputs*max_n_classes
        self.max_n_classes = max_n_classes

        free(self.errors)
        self.error_size = n_instances*self.inst_stride
        self.predictions_size = self.error_size
        self.errors = <double*> calloc(self.error_size, sizeof(double))
        self.predictions = self.errors

        free(self.current_weights)
        self.weights_size =  max_n_classes*n_outputs
        self.current_weights = <double*> calloc(self.weights_size, sizeof(double))

        free(self.buffer)
        self.buffer = <double*> calloc(self.weights_size, sizeof(double))


        # Compute the error for the null model
        cdef SIZE_t i, predictions_size = self.predictions_size
        cdef double* predictions = self.predictions
        for i in range(predictions_size):
            predictions[i] = 0




    def proba_transformer(self):
        def TODO(raw_pred):
            return raw_pred
        return



    cdef void update_errors(self, SIZE_t start, SIZE_t end, SIZE_t* indices,
                            double* deltas) nogil:

        # deltas is [n_outputs, n_classes]
        # Both self.predictions and self.y are [n_instances, n_outputs]

        cdef double* predictions = self.predictions
        cdef double* y = self.y
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_classes = self.max_n_classes
        cdef SIZE_t n_effective_classes = self.max_n_classes
        cdef SIZE_t inst_stride = self.inst_stride
        cdef SIZE_t out_stride = self.out_stride
        cdef SIZE_t cls_stride = self.cls_stride

        cdef SIZE_t inst, out, cls, idx, index, deltas_idx



        for inst in range(start, end):
            for out in range(n_outputs):
                for cls in range(n_classes):
                    deltas_idx = out*n_classes + cls
                    if deltas[deltas_idx]  <= -DBL_MAX:
                        continue
                    index = indices[inst]*inst_stride
                    predictions[index*inst_stride+
                                out*out_stride+
                                cls*cls_stride] += deltas[deltas_idx]





    cdef double optimize_weight(self,
                                SIZE_t* indices,
                                SIZE_t start,
                                SIZE_t end) nogil:

        return -1


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
