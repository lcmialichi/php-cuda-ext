#include <cuda_runtime.h>
#include "unary_ops.h"
#include "unary_ops.cuh"
#include "new_ops_func.cuh"
#include "dispatcher.h"

extern "C" void launch_unary_op(
    void *base,
    void *result,
    size_t base_offset,
    dtype_t result_dtype,
    operation_type_t op_type,
    int *shape,
    size_t *strides,
    int ndims,
    size_t total_size)
{
    DISPATCH_DTYPE(result_dtype, {
        DISPATCH_UNARY_OP(op_type, {
            launch_unary_op_kernel<scalar_t, bin_op_t>(
                (scalar_t *)base,
                (scalar_t *)result,
                base_offset,
                shape,
                strides,
                ndims,
                total_size);
        });
    });
}
