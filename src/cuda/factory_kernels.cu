#include <cuda_runtime.h>
#include "factory_kernels.cuh"
#include "dispatcher.h"
#include "../data_types.h"

extern "C" void launch_assign_scalar_val_kernel(
    void *base,
    dtype_t dtype,
    scalar_value_t value,
    size_t total_elements)
{
    DISPATCH_DTYPE(dtype, {
        scalar_t scalar;

        switch (dtype)
        {
        case DTYPE_FLOAT32:
            scalar = (scalar_t)value.v.f32;
            break;
        case DTYPE_FLOAT64:
            scalar = (scalar_t)value.v.f64;
            break;
        case DTYPE_INT32:
            scalar = (scalar_t)value.v.i32;
            break;
        case DTYPE_INT64:
            scalar = (scalar_t)value.v.i64;
            break;
        case DTYPE_INT8:
            scalar = (scalar_t)value.v.i8;
            break;
        case DTYPE_BOOL:
            scalar = (scalar_t)value.v.b;
            break;
        default:
            scalar = (scalar_t)0;
            break;
        }

        launch_fill_kernel_with_scalar<scalar_t>(
            (scalar_t *)base,
            scalar,
            total_elements);
    });
}
