#ifndef UNARY_OPS_CUH
#define UNARY_OPS_CUH

#include <cuda_runtime.h>

#define MAX_DIMS 10
struct UnaryParams {
    int shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    int ndims;
};

__constant__ UnaryParams d_unary_params;

template <typename Op>
__global__ void unary_kernel_strided(
    const float *base,
    float *result,
    size_t base_offset,
    size_t total_size)
{
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    size_t offset = 0;
    size_t remaining = idx;

    for (int d = d_unary_params.ndims - 1; d >= 0; d--)
    {
        size_t coord = remaining % d_unary_params.shape[d];
        remaining /= d_unary_params.shape[d];
        offset += coord * d_unary_params.strides[d];
    }

    Op op;
    result[base_offset + offset] = op(base[base_offset + offset]);
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

    UnaryParams h_params;
    memcpy(h_params.shape, shape, ndims * sizeof(int));
    memcpy(h_params.strides, strides, ndims * sizeof(size_t));
    h_params.ndims = ndims;

    cudaMemcpyToSymbol(d_unary_params, &h_params, sizeof(UnaryParams));
    unary_kernel_strided<Op><<<blocks, threads>>>(
        base,
        result,
        base_offset,
        total_size);
}

#endif
