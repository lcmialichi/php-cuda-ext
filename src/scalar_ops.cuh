#ifndef SCALAR_OPS_CUH
#define SCALAR_OPS_CUH

#include <cuda_runtime.h>

template <typename Op>
__global__ void scalar_kernel(const float *a, float scalar, float *result, size_t total)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    Op op;
    result[idx] = op(a[idx], scalar);
}

template <typename Op>
void launch_scalar_op(float *a, float scalar, float *result, size_t total)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scalar_kernel<Op><<<blocks, threads>>>(a, scalar, result, total);
}

#endif
