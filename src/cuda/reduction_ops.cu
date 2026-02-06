#include <cuda_runtime.h>
#include "reduction_ops.cuh"
#include "reduction_ops.h"
#include "dispatcher.h"
#include <stdio.h>

extern "C" void launch_reduction(
    void *input, void *output,
    dtype_t dtype, operation_type_t op_type,
    int *input_shape, int input_ndims,
    int *result_shape,
    size_t *input_strides,
    int result_ndims, int axis,
    size_t total_elements_out, size_t input_base_offset)
{
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP_REDUCTION(op_type, {
            launch_reduce_op_kernel<scalar_t, bin_op_t>(
                (scalar_t *)input,
                (scalar_t *)output,
                input_shape,
                input_ndims,
                result_shape,
                input_strides,
                result_ndims,
                axis,
                total_elements_out,
                input_base_offset);
        });
    });
}

extern "C" void launch_arg_reduction(
    void *input, int *output,
    dtype_t dtype, operation_type_t op_type,
    int *input_shape, int input_ndims,
    int *result_shape,
    size_t *input_strides,
    int result_ndims, int axis,
    size_t total_elements_out, size_t input_base_offset)
{
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP_ARG_REDUCTION(op_type, {
            launch_arg_reduce_kernel<scalar_t, bin_op_t>(
                (scalar_t *)input,
                output,
                input_shape,
                input_ndims,
                input_strides,
                axis,
                total_elements_out,
                input_base_offset);
        });
    });
}
