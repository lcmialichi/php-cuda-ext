#ifndef REDUCTION_OPS_H
#define REDUCTION_OPS_H

#include <stddef.h>
#include "../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void (*reduction_fn)(float *input, float *result, int *input_shape, int input_ndims, int *result_shape, size_t *input_strides, int result_ndims, int axis, size_t total_elements, size_t input_base_offset);
    typedef struct
    {
        operation_type_t op;
        reduction_fn fn;
    } ReductionDispatchEntry;

    typedef void (*reduction_arg_fn)(float *input, int *result_idx,
                                     int *input_shape, int input_ndims,
                                     size_t *input_strides,
                                     int axis,
                                     size_t total_elements, size_t input_base_offset);
    typedef struct
    {
        int op;
        reduction_arg_fn fn;
    } ReductionArgDispatchEntry;

    void launch_reduce_sum(float *input, float *result, int *input_shape, int input_ndims, int *result_shape, size_t *input_strides, int result_ndims, int axis, size_t total_elements, size_t input_base_offset);
    void launch_reduce_mean(float *input, float *result, int *input_shape, int input_ndims, int *result_shape, size_t *input_strides, int result_ndims, int axis, size_t total_elements, size_t input_base_offset);
    void launch_reduce_max(float *input, float *result, int *input_shape, int input_ndims, int *result_shape, size_t *input_strides, int result_ndims, int axis, size_t total_elements, size_t input_base_offset);
    void launch_reduce_min(float *input, float *result, int *input_shape, int input_ndims, int *result_shape, size_t *input_strides, int result_ndims, int axis, size_t total_elements, size_t input_base_offset);
    void launch_reduce_prod(float *input, float *result, int *input_shape, int input_ndims, int *result_shape, size_t *input_strides, int result_ndims, int axis, size_t total_elements, size_t input_base_offset);
    void launch_arg_max(float *input, int *result_idx,
                        int *input_shape, int input_ndims,
                        size_t *input_strides,
                        int axis,
                        size_t total_elements, size_t input_base_offset);

    void launch_arg_min(float *input, int *result_idx,
                        int *input_shape, int input_ndims,
                        size_t *input_strides,
                        int axis,
                        size_t total_elements, size_t input_base_offset);

    extern ReductionDispatchEntry reduction_dispatch[];
#ifdef __cplusplus
}
#endif
#endif