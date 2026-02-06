#include <cuda_runtime.h>
#include "scalar_ops.h"
#include "scalar_ops.cuh"
#include "dispatcher.h"
#include "../data_types.h"

extern "C" void launch_scalar(
    void *base,
    dtype_t base_dtype,
    scalar_value_t scalar_val,
    void *result,
    dtype_t result_dtype,
    operation_type_t op_type,
    size_t base_offset,
    int *shape,
    size_t *strides,
    int ndims,
    size_t total_size,
    int is_contiguous
)
{
    DISPATCH_DTYPE(result_dtype, {
        scalar_t val = convert_union_to_type<scalar_t>(scalar_val);
        DISPATCH_OP(op_type, {
            launch_scalar_op<scalar_t, bin_op_t>(
                base,
                base_dtype,
                val, 
                (scalar_t *)result, 
                base_offset, 
                shape, 
                strides, 
                ndims, 
                total_size,
                is_contiguous
            );
        });
    });
}

extern "C" void launch_scalar_inv(
    void *base,
    dtype_t base_dtype,
    scalar_value_t scalar_val,
    void *result,
    dtype_t result_dtype,
    operation_type_t op_type,
    size_t base_offset,
    int *shape,
    size_t *strides,
    int ndims,
    size_t total_size,
    int is_contiguous
)
{
    DISPATCH_DTYPE(result_dtype, {
        scalar_t val = convert_union_to_type<scalar_t>(scalar_val);
        DISPATCH_OP(op_type, {
            launch_inv_scalar_op<scalar_t, bin_op_t>(
                base,
                base_dtype,
                val, 
                (scalar_t *)result, 
                base_offset, 
                shape, 
                strides, 
                ndims, 
                total_size,
                is_contiguous
            );
        });
    });
}
