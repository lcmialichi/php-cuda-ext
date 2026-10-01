#include "concat_kernels.h"
#include "launch_config.cuh"

struct ConcatParams
{
    void *input_ptrs[MAX_CONCAT_TENSORS];
    int input_axis_sizes[MAX_CONCAT_TENSORS];
    size_t outer_dims;
    size_t inner_dims;
    int output_axis_size;
    int num_tensors;
};

template <typename T>
static __global__ void concat_kernel(ConcatParams params, T *output)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = params.outer_dims * params.output_axis_size * params.inner_dims;
    if (idx >= total)
        return;

    size_t outer_idx = idx / (params.output_axis_size * params.inner_dims);
    size_t axis_idx = (idx / params.inner_dims) % params.output_axis_size;
    size_t inner_idx = idx % params.inner_dims;

    int tensor_idx = 0;
    size_t axis_offset = 0;
    for (; tensor_idx < params.num_tensors; tensor_idx++)
    {
        size_t axis_size = params.input_axis_sizes[tensor_idx];
        if (axis_idx < axis_offset + axis_size)
            break;
        axis_offset += axis_size;
    }

    if (tensor_idx >= params.num_tensors)
        return;

    const T *input = (const T *)params.input_ptrs[tensor_idx];
    size_t input_offset = outer_idx * params.input_axis_sizes[tensor_idx] * params.inner_dims;
    input_offset += (axis_idx - axis_offset) * params.inner_dims + inner_idx;
    output[idx] = input[input_offset];
}

extern "C" int launch_concat_kernel_host(
    tensor_t **input_tensors, int num_tensors, tensor_t *output_tensor, int axis,
    size_t outer_dims, size_t inner_dims, int output_axis_size)
{
    if (num_tensors < 1 || num_tensors > MAX_CONCAT_TENSORS)
        return 1;

    size_t total_elements = outer_dims * output_axis_size * inner_dims;
    if (total_elements == 0)
        return 0;

    ConcatParams params = {};
    params.num_tensors = num_tensors;
    params.outer_dims = outer_dims;
    params.inner_dims = inner_dims;
    params.output_axis_size = output_axis_size;

    for (int i = 0; i < num_tensors; i++)
    {
        params.input_ptrs[i] = input_tensors[i]->data;
        params.input_axis_sizes[i] = input_tensors[i]->shape[axis];
    }

    switch (output_tensor->dtype)
    {
    case DTYPE_FLOAT32:
        concat_kernel<float><<<cuda_grid_1d(total_elements), 256>>>(params, (float *)output_tensor->data);
        break;
    case DTYPE_INT32:
        concat_kernel<int><<<cuda_grid_1d(total_elements), 256>>>(params, (int *)output_tensor->data);
        break;
    default:
        return 1;
    }

    if (cuda_launch_status(true) != cudaSuccess)
        return 1;
    return 0;
}