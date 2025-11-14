#ifndef BROADCAST_OPS_CUH
#define BROADCAST_OPS_CUH

#include <cuda_runtime.h>
#include <vector>

template <typename Op>
__global__ void broadcast_kernel_opt(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ result,
    const int *__restrict__ a_strides,
    const int *__restrict__ b_strides,
    const int *__restrict__ shape,
    int dims,
    size_t total)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    size_t tmp = idx;
    int a_idx = 0;
    int b_idx = 0;

#pragma unroll 8
    for (int i = dims - 1; i >= 0; i--)
    {
        int coord = tmp % shape[i];
        tmp /= shape[i];

        a_idx += coord * a_strides[i];
        b_idx += coord * b_strides[i];
    }

    Op op;
    result[idx] = op(a[a_idx], b[b_idx]);
}

template <typename Op>
void launch_broadcast_op(float *a, float *b, float *result,
                         int *a_strides, int a_dims,
                         int *b_strides, int b_dims,
                         int *result_shape, int result_dims,
                         size_t total_elements)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    std::vector<int> a_strides_full(result_dims, 0);
    std::vector<int> b_strides_full(result_dims, 0);

    for (int i = 0; i < a_dims; i++)
        a_strides_full[result_dims - a_dims + i] = a_strides[i];

    for (int i = 0; i < b_dims; i++)
        b_strides_full[result_dims - b_dims + i] = b_strides[i];

    int *d_a_strides, *d_b_strides, *d_shape;

    cudaMalloc(&d_a_strides, result_dims * sizeof(int));
    cudaMalloc(&d_b_strides, result_dims * sizeof(int));
    cudaMalloc(&d_shape, result_dims * sizeof(int));

    cudaMemcpy(d_a_strides, a_strides_full.data(),
               result_dims * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_b_strides, b_strides_full.data(),
               result_dims * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_shape, result_shape,
               result_dims * sizeof(int), cudaMemcpyHostToDevice);

    broadcast_kernel_opt<Op><<<blocks, threads>>>(
        a, b, result,
        d_a_strides,
        d_b_strides,
        d_shape,
        result_dims,
        total_elements);

    cudaFree(d_a_strides);
    cudaFree(d_b_strides);
    cudaFree(d_shape);
}

#endif
