#ifndef FACTORY_KERNELS_CUH
#define FACTORY_KERNELS_CUH
#include "../data_types.h"

template <typename T>
static __host__ T get_scalar_value(scalar_value_t s)
{
    if (std::is_same<T, float>::value)
        return (T)s.v.f32;
    if (std::is_same<T, double>::value)
        return (T)s.v.f64;
    if (std::is_same<T, int32_t>::value)
        return (T)s.v.i32;
    if (std::is_same<T, int64_t>::value)
        return (T)s.v.i64;
    if (std::is_same<T, int8_t>::value)
        return (T)s.v.i8;
    if (std::is_same<T, bool>::value)
        return (T)s.v.b;
    return (T)0;
}

template <typename T>
__global__ void assign_scalar_val_kernel(
    T *__restrict__ base,
    T scalar,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        base[idx] = (T)scalar;
    }
}

template <typename T>
void launch_fill_kernel_with_scalar(T *base, T scalar, size_t total_elements)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535)
        blocks = 65535;

    assign_scalar_val_kernel<T><<<blocks, threads>>>(
        base, scalar, total_elements);
}

#endif