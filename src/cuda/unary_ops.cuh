#ifndef UNARY_OPS_CUH
#define UNARY_OPS_CUH

#include <cuda_runtime.h>

template <typename Op>
__global__ void unary_kernel_strided(
    const float *base,
    float *result,
    size_t base_offset,
    const int *shape,
    const size_t *strides,
    int ndims,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    size_t offset = 0;
    size_t remaining = idx;

    for (int d = ndims - 1; d >= 0; d--)
    {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        offset += coord * strides[d];
    }

    Op op;
    result[base_offset + offset] = op(base[offset]);
}

template <typename Op>
void launch_unary_op(
    float *base,
    float *result,
    size_t base_offset,
    int *shape,
    size_t *strides,
    int ndims,
    size_t total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    int *d_shape;
    size_t *d_strides;

    cudaMalloc(&d_shape, ndims * sizeof(int));
    cudaMalloc(&d_strides, ndims * sizeof(size_t));
    cudaMemcpy(d_shape, shape, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides, ndims * sizeof(size_t), cudaMemcpyHostToDevice);

    unary_kernel_strided<Op><<<blocks, threads>>>(
        base,
        result,
        base_offset,
        d_shape,
        d_strides,
        ndims,
        total_size);

    cudaFree(d_shape);
    cudaFree(d_strides);
}

#endif
