#ifndef BROADCAST_OPS_CUH
#define BROADCAST_OPS_CUH

#include <cuda_runtime.h>
#include <vector>

#define MAX_DIMS 10

struct BroadcastParams
{
    int a_strides_full[MAX_DIMS];
    int b_strides_full[MAX_DIMS];
    int shape[MAX_DIMS];
    int dims;
};

__constant__ BroadcastParams d_params;

template <typename Op>
__global__ void broadcast_kernel_opt(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ result,
    size_t total,
    size_t a_base_offset,
    size_t b_base_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    size_t tmp = idx;
    int a_idx = 0;
    int b_idx = 0;

#pragma unroll 8
    for (int i = d_params.dims - 1; i >= 0; i--)
    {
        int coord = tmp % d_params.shape[i];
        tmp /= d_params.shape[i];

        a_idx += coord * d_params.a_strides_full[i];
        b_idx += coord * d_params.b_strides_full[i];
    }

    Op op;
    result[idx] = op(a[a_idx + a_base_offset], b[b_idx + b_base_offset]);
}

template <typename Op>
void launch_broadcast_op(float *a, float *b, float *result,
                         int *a_strides, int a_dims,
                         int *b_strides, int b_dims,
                         int *result_shape, int result_dims,
                         size_t total_elements,
                         size_t a_base_offset,
                         size_t b_base_offset)
{
    int threads = 256;
    int blocks = min(32, (int)((total_elements + threads - 1) / threads));

    std::vector<int> a_strides_full(result_dims, 0);
    std::vector<int> b_strides_full(result_dims, 0);

    for (int i = 0; i < result_dims; i++)
    {
        int a_index = i - (result_dims - a_dims);

        if (a_index >= 0)
        {
            a_strides_full[i] = a_strides[a_index];
        }
        else
        {
            a_strides_full[i] = 0;
        }
    }

    for (int i = 0; i < result_dims; i++)
    {
        int b_index = i - (result_dims - b_dims);

        if (b_index >= 0)
        {
            b_strides_full[i] = b_strides[b_index];
        }
        else
        {
            b_strides_full[i] = 0;
        }
    }

    BroadcastParams h_params;
    memcpy(h_params.a_strides_full, a_strides_full.data(), result_dims * sizeof(int));
    memcpy(h_params.b_strides_full, b_strides_full.data(), result_dims * sizeof(int));
    memcpy(h_params.shape, result_shape, result_dims * sizeof(int));
    h_params.dims = result_dims;

    cudaMemcpyToSymbol(d_params, &h_params, sizeof(BroadcastParams));
    broadcast_kernel_opt<Op><<<blocks, threads>>>(
        a, b, result,
        total_elements,
        a_base_offset,
        b_base_offset);
}

#endif
