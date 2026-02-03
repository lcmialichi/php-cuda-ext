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

extern "C" void launch_scale_range_kernel(
    float *values,
    void *base,
    dtype_t dtype,
    scalar_value_t min,
    scalar_value_t max,
    size_t total_elements)
{
    if (total_elements == 0) return;

    DISPATCH_DTYPE(dtype, {
        scalar_t s_min, s_max;
        
        switch (dtype) {
            case DTYPE_FLOAT32: s_min = (scalar_t)min.v.f32; s_max = (scalar_t)max.v.f32; break;
            case DTYPE_FLOAT64: s_min = (scalar_t)min.v.f64; s_max = (scalar_t)max.v.f64; break;
            case DTYPE_INT32:   s_min = (scalar_t)min.v.i32; s_max = (scalar_t)max.v.i32; break;
            case DTYPE_INT64:   s_min = (scalar_t)min.v.i64; s_max = (scalar_t)max.v.i64; break;
            case DTYPE_INT8:    s_min = (scalar_t)min.v.i8;  s_max = (scalar_t)max.v.i8;  break;
            case DTYPE_BOOL:    s_min = (scalar_t)min.v.b;   s_max = (scalar_t)max.v.b;   break;
            default:            s_min = (scalar_t)0;         s_max = (scalar_t)0;         break;
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

        scale_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            values,
            (scalar_t *)base,
            total_elements,
            s_min,
            s_max
        );
    });
}
