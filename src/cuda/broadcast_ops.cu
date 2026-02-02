#include <cuda_runtime.h>
#include "new_ops_func.cuh"
#include "broadcast_ops.cuh"
#include "broadcast_ops.h"
#include "dispatcher.h"

extern "C" void launch_broadcast(
    void *a, dtype_t dtype_a, void *b, dtype_t dtype_b, void *result,
    dtype_t dtype, operation_type_t op_type,
    int *a_strides, int a_dims,
    int *b_strides, int b_dims,
    int *result_shape, int result_dims,
    size_t total_elements, size_t a_offset, size_t b_offset)
{
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP(op_type, {
            launch_broadcast_kernel<scalar_t, bin_op_t>(
                a, dtype_a,
                b, dtype_b,
                (scalar_t *)result,
                a_strides, a_dims,
                b_strides, b_dims,
                result_shape, result_dims,
                total_elements,
                a_offset,
                b_offset);
        });
    });
}