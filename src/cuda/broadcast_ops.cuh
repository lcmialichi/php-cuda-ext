#ifndef BROADCAST_OPS_CUH
#define BROADCAST_OPS_CUH

#include <cuda_runtime.h>
#include "../data_types.h"
#include "cast.cuh"

#define MAX_DIMS 10

struct BroadcastParams {
    size_t a_strides[MAX_DIMS];
    size_t b_strides[MAX_DIMS];
    size_t shape[MAX_DIMS];
    int dims;
    bool is_contiguous;
};

template <typename T, typename Op>
__global__ void broadcast_kernel(
    const void *__restrict__ a,
    const dtype_t dtype_a,
    const void *__restrict__ b,
    const dtype_t dtype_b,
    T *__restrict__ result,
    const size_t total,
    const BroadcastParams params, 
    const size_t a_base_offset,
    const size_t b_base_offset) 
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_idx = 0;
    size_t b_idx = 0;

    if (params.is_contiguous) {
        a_idx = idx;
        b_idx = idx;
    } else {
        size_t tmp = idx;
        #pragma unroll
        for (int i = params.dims - 1; i >= 0; --i) {
            const size_t coord = tmp % params.shape[i];
            tmp /= params.shape[i];
            a_idx += coord * params.a_strides[i];
            b_idx += coord * params.b_strides[i];
        }
    }

    const T val_a = fetch_and_cast<T>(a, dtype_a, a_idx + a_base_offset);
    const T val_b = fetch_and_cast<T>(b, dtype_b, b_idx + b_base_offset);

    result[idx] = Op::apply(val_a, val_b);
}

template <typename T, typename Op>
void launch_broadcast_kernel(void *a, dtype_t dtype_a, void *b, dtype_t dtype_b, T *result,
                             int *a_strides, int a_dims,
                             int *b_strides, int b_dims,
                             int *result_shape, int result_dims,
                             size_t total_elements,
                             size_t a_base_offset,
                             size_t b_base_offset) 
{
    BroadcastParams h_params;
    h_params.dims = result_dims;
    h_params.is_contiguous = (a_dims == result_dims && b_dims == result_dims);

    for (int i = 0; i < result_dims; i++) {
        int a_offset = i - (result_dims - a_dims);
        int b_offset = i - (result_dims - b_dims);
        
        h_params.a_strides[i] = (a_offset >= 0) ? a_strides[a_offset] : 0;
        h_params.b_strides[i] = (b_offset >= 0) ? b_strides[b_offset] : 0;
        h_params.shape[i] = result_shape[i];

        if (h_params.a_strides[i] == 0 || h_params.b_strides[i] == 0) h_params.is_contiguous = false;
    }

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    broadcast_kernel<T, Op><<<blocks, threads>>>(
        a, dtype_a, b, dtype_b, result,
        total_elements,
        h_params,
        a_base_offset,
        b_base_offset);
}

#endif