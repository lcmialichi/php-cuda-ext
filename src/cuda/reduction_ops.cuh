#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include "reduction_ops.h"

#define MAX_DIMS 10

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

template <typename Op, typename Identity>
__global__ void reduce_kernel(
    const float *__restrict__ input,
    float *__restrict__ result,
    size_t input_base_offset)
{
    size_t idx_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_out >= d_reduce_params.total_elements_out)
        return;

    int coords[MAX_DIMS];
    size_t temp_idx = idx_out;

    int non_reduced_axes_map[MAX_DIMS];
    int output_shape_map[MAX_DIMS];
    int output_ndims = 0;

    for (int i = 0; i < d_reduce_params.ndims; ++i)
    {
        if (i != d_reduce_params.reduce_axis)
        {
            output_shape_map[output_ndims] = d_reduce_params.d_shape[i];
            non_reduced_axes_map[output_ndims] = i;
            output_ndims++;
        }
    }

    for (int i = output_ndims - 1; i >= 0; --i)
    {
        coords[non_reduced_axes_map[i]] = temp_idx % output_shape_map[i];
        temp_idx /= output_shape_map[i];
    }

    coords[d_reduce_params.reduce_axis] = 0;

    Op op;
    Identity identity;
    float accumulator = identity();

    int reduce_dim_size = d_reduce_params.d_shape[d_reduce_params.reduce_axis];
    
    for (int current_idx = 0; current_idx < reduce_dim_size; ++current_idx)
    {
        coords[d_reduce_params.reduce_axis] = current_idx;
        size_t flat_index = get_linear_index(coords);
        float current_val = input[flat_index];

        accumulator = op(accumulator, current_val);
    }

    result[idx_out] = accumulator;
}

template <typename Op>
__global__ void arg_reduce_kernel(
    const float *__restrict__ input,
    int *__restrict__ result_idx)
{
    size_t idx_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_out >= d_reduce_params.total_elements_out)
        return;

    int coords[MAX_DIMS];
    size_t temp_idx = idx_out;
    int non_reduced_axes_map[MAX_DIMS];
    int output_shape_map[MAX_DIMS];
    int output_ndims = 0;

    for (int i = 0; i < d_reduce_params.ndims; ++i)
    {
        if (i != d_reduce_params.reduce_axis)
        {
            output_shape_map[output_ndims] = (int)d_reduce_params.d_shape[i];
            non_reduced_axes_map[output_ndims] = i;
            output_ndims++;
        }
    }

    for (int i = output_ndims - 1; i >= 0; --i)
    {
        coords[non_reduced_axes_map[i]] = temp_idx % output_shape_map[i];
        temp_idx /= output_shape_map[i];
    }

    coords[d_reduce_params.reduce_axis] = 0;

    int best_idx = 0;
    size_t start_flat_index = get_linear_index(coords);
    float best_val = input[start_flat_index];

    Op op;

    int reduce_dim_size = d_reduce_params.d_shape[d_reduce_params.reduce_axis];

    for (int current_idx = 1; current_idx < reduce_dim_size; ++current_idx)
    {
        coords[d_reduce_params.reduce_axis] = current_idx;
        size_t flat_index = get_linear_index(coords);
        float current_val = input[flat_index];

        if (op(current_val, best_val))
        {
            best_val = current_val;
            best_idx = current_idx;
        }
    }
    result_idx[idx_out] = best_idx;
}

template <typename Op, typename Identity>
void launch_reduce_op(float *input, float *result,
                      int *input_shape, int input_ndims,
                      int *result_shape, size_t *input_strides, int result_ndims,
                      int axis,
                      size_t total_elements, size_t input_base_offset)
{
    if (total_elements == 0)
        return;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    ReductionParams h_params;
    memcpy(h_params.d_shape, input_shape, input_ndims * sizeof(int));
    memcpy(h_params.d_strides, input_strides, input_ndims * sizeof(size_t));
    h_params.ndims = input_ndims;
    h_params.reduce_axis = axis;
    h_params.reduce_dim_size = input_shape[axis];
    h_params.total_elements_out = total_elements;

    cudaMemcpyToSymbol(d_reduce_params, &h_params, sizeof(ReductionParams));

    reduce_kernel<Op, Identity><<<blocks, threads>>>(
        input, result, input_base_offset);
}

template <typename Op>
void launch_arg_reduce(float *input, int *result_idx,
                       int *input_shape, int input_ndims,
                       size_t *input_strides,
                       int axis,
                       size_t total_elements, size_t input_base_offset)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    ReductionParams h_params;
    memcpy(h_params.d_shape, input_shape, input_ndims * sizeof(int));
    memcpy(h_params.d_strides, input_strides, input_ndims * sizeof(size_t));

    h_params.ndims = input_ndims;
    h_params.reduce_axis = axis;
    h_params.total_elements_out = total_elements;

    cudaMemcpyToSymbol(d_reduce_params, &h_params, sizeof(ReductionParams));

    arg_reduce_kernel<Op><<<blocks, threads>>>(input, result_idx);
}
