#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include <string.h>
#include <float.h>
#include <cstdint>
#include <cmath>

#define MAX_DIMS 10
#define REDUCTION_BLOCK_SIZE 256

struct ReductionParams
{
    int ndims;
    int reduce_axis;
    size_t total_elements_out;
    size_t reduce_dim_size;
    int d_shape[MAX_DIMS];
    size_t d_strides[MAX_DIMS];
};

__constant__ ReductionParams d_reduce_params;

__device__ size_t get_linear_index(const int *coords)
{
    size_t index = 0;
    for (int i = 0; i < d_reduce_params.ndims; ++i)
    {
        index += (size_t)coords[i] * d_reduce_params.d_strides[i];
    }
    return index;
}

template <typename Op>
__global__ void reduce_kernel_optimized(
    const float *__restrict__ input,
    float *__restrict__ result,
    size_t input_base_offset)
{
    extern __shared__ float sdata[];

    Op op;
    ArgIdentity<Op> arg_identity;

    size_t idx_out = blockIdx.x;
    if (idx_out >= d_reduce_params.total_elements_out)
        return;

    int tid = threadIdx.x;
    int reduce_dim_size = d_reduce_params.reduce_dim_size;

    int coords[MAX_DIMS] = {0};
    size_t temp_idx = idx_out;

    for (int i = d_reduce_params.ndims - 1; i >= 0; --i)
    {
        if (i != d_reduce_params.reduce_axis)
        {
            coords[i] = temp_idx % d_reduce_params.d_shape[i];
            temp_idx /= d_reduce_params.d_shape[i];
        }
    }

    float accumulator = arg_identity.get_init_val();

    for (int current_idx = tid; current_idx < reduce_dim_size; current_idx += blockDim.x)
    {
        coords[d_reduce_params.reduce_axis] = current_idx;
        size_t flat_index = get_linear_index(coords) + input_base_offset;

        float current_val = input[flat_index];
        accumulator = op(accumulator, current_val);
    }

    sdata[tid] = accumulator;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile float *vsdata = sdata;
        for (int s = 16; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                vsdata[tid] = op(vsdata[tid], vsdata[tid + s]);
            }
        }
    }

    if (tid == 0)
    {
        result[idx_out] = sdata[0];
    }
}

template <typename Op, int BLOCK_SIZE>
__global__ void reduce_contiguous_fast(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t reduction_size,
    size_t outer_size)
{
    const int tid = threadIdx.x;
    const int outer_idx = blockIdx.x;

    if (outer_idx >= outer_size)
        return;

    Op op;
    ArgIdentity<Op> identity;

    __shared__ float smem[BLOCK_SIZE];

    float thread_val = identity.get_init_val();
    const size_t base_idx = outer_idx * reduction_size;

    for (size_t i = tid; i < reduction_size; i += BLOCK_SIZE)
    {
        thread_val = op(thread_val, input[base_idx + i]);
    }

    smem[tid] = thread_val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] = op(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[outer_idx] = smem[0];
    }
}

template <typename Op>
__global__ void arg_reduce_optimized(
    const float *__restrict__ input,
    int *__restrict__ result_idx,
    size_t input_base_offset)
{
    extern __shared__ float sdata_vals_and_indices[];
    float *sdata_vals = sdata_vals_and_indices;
    int *sdata_indices = (int *)&sdata_vals_and_indices[blockDim.x];

    Op op;
    ArgIdentity<Op> arg_identity;

    size_t idx_out = blockIdx.x;
    if (idx_out >= d_reduce_params.total_elements_out)
        return;

    int tid = threadIdx.x;
    int reduce_dim_size = d_reduce_params.reduce_dim_size;

    int coords[MAX_DIMS] = {0};
    size_t temp_idx = idx_out;

    for (int i = d_reduce_params.ndims - 1; i >= 0; --i)
    {
        if (i != d_reduce_params.reduce_axis)
        {
            coords[i] = temp_idx % d_reduce_params.d_shape[i];
            temp_idx /= d_reduce_params.d_shape[i];
        }
    }

    float best_val = arg_identity.get_init_val();
    int best_idx = -1;

    if (tid < reduce_dim_size)
    {
        coords[d_reduce_params.reduce_axis] = tid;
        size_t flat_index = get_linear_index(coords) + input_base_offset;
        best_val = input[flat_index];
        best_idx = tid;
    }

    for (int current_idx = tid + blockDim.x; current_idx < reduce_dim_size; current_idx += blockDim.x)
    {
        coords[d_reduce_params.reduce_axis] = current_idx;
        size_t flat_index = get_linear_index(coords) + input_base_offset;
        float current_val = input[flat_index];

        if (op(current_val, best_val))
        {
            best_val = current_val;
            best_idx = current_idx;
        }
    }

    sdata_vals[tid] = best_val;
    sdata_indices[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (op(sdata_vals[tid + s], sdata_vals[tid]))
            {
                sdata_vals[tid] = sdata_vals[tid + s];
                sdata_indices[tid] = sdata_indices[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        result_idx[idx_out] = sdata_indices[0];
    }
}

template <typename Op>
void launch_reduce_op(float *input, float *result,
                      int *input_shape, int input_ndims,
                      int *result_shape, size_t *input_strides, int result_ndims,
                      int axis,
                      size_t total_elements_out, size_t input_base_offset)
{
    if (total_elements_out == 0)
        return;

    ReductionParams h_params;
    h_params.ndims = input_ndims;
    h_params.reduce_axis = axis;
    h_params.total_elements_out = total_elements_out;
    h_params.reduce_dim_size = input_shape[axis];

    memcpy(h_params.d_shape, input_shape, input_ndims * sizeof(int));
    memcpy(h_params.d_strides, input_strides, input_ndims * sizeof(size_t));

    cudaMemcpyToSymbol(d_reduce_params, &h_params, sizeof(ReductionParams));

    int threads = REDUCTION_BLOCK_SIZE;
    int blocks = total_elements_out;
    size_t shared_mem_size = threads * sizeof(float);

    reduce_kernel_optimized<Op><<<blocks, threads, shared_mem_size>>>(
        input, result, input_base_offset);
}

template <typename Op>
void launch_arg_reduce(float *input, int *result_idx,
                       int *input_shape, int input_ndims,
                       size_t *input_strides,
                       int axis,
                       size_t total_elements_out, size_t input_base_offset)
{
    if (total_elements_out == 0)
        return;

    ReductionParams h_params;
    h_params.ndims = input_ndims;
    h_params.reduce_axis = axis;
    h_params.total_elements_out = total_elements_out;
    h_params.reduce_dim_size = input_shape[axis];

    memcpy(h_params.d_shape, input_shape, input_ndims * sizeof(int));
    memcpy(h_params.d_strides, input_strides, input_ndims * sizeof(size_t));

    cudaMemcpyToSymbol(d_reduce_params, &h_params, sizeof(ReductionParams));

    int threads = REDUCTION_BLOCK_SIZE;
    int blocks = total_elements_out;
    size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));

    arg_reduce_optimized<Op><<<blocks, threads, shared_mem_size>>>(
        input, result_idx, input_base_offset);
}