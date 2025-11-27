#include <cuda_runtime.h>
#include <math.h>
#include "cuda_kernels.h"
#include <float.h>
#include <cstdio>

#define SUCCESS 0
#define FAILURE 1

extern "C"
{
    __global__ void fill_kernel(float *data, float value, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            data[idx] = value;
    }

    __global__ void scale_kernel(float *data, size_t size, float min_value, float max_value)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size)
        {
            float raw_rand = data[idx];
            float range = max_value - min_value;

            data[idx] = min_value + range * raw_rand;
        }
    }

    __global__ void clip_kernel(float *a, float min_val, float max_val, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = fminf(fmaxf(a[i], min_val), max_val);
        }
    }

    __global__ void relu_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = fmaxf(a[i], 0.0f);
        }
    }

    __global__ void sigmoid_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = 1.0f / (1.0f + expf(-a[i]));
        }
    }

    __global__ void tanh_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = tanhf(a[i]);
        }
    }

    __global__ void concat_kernel_int32(ConcatParams *params, int *output)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= params->outer_dims * params->output_axis_size * params->inner_dims)
        {
            return;
        }

        size_t outer_idx = idx / (params->output_axis_size * params->inner_dims);
        size_t axis_idx = (idx / params->inner_dims) % params->output_axis_size;
        size_t inner_idx = idx % params->inner_dims;

        int tensor_idx = 0;
        size_t current_offset = 0;

        for (tensor_idx = 0; tensor_idx < params->num_tensors; tensor_idx++)
        {
            size_t tensor_axis_size = params->input_axis_sizes[tensor_idx];
            if (axis_idx < current_offset + tensor_axis_size)
            {
                break;
            }
            current_offset += tensor_axis_size;
        }

        if (tensor_idx >= params->num_tensors)
        {
            return;
        }

        size_t local_axis_idx = axis_idx - current_offset;
        int *input_tensor = (int *)params->input_ptrs[tensor_idx];

        size_t input_offset = outer_idx * params->input_axis_sizes[tensor_idx] * params->inner_dims;
        input_offset += local_axis_idx * params->inner_dims;
        input_offset += inner_idx;

        size_t output_offset = outer_idx * params->output_axis_size * params->inner_dims;
        output_offset += axis_idx * params->inner_dims;
        output_offset += inner_idx;

        output[output_offset] = input_tensor[input_offset];
    }

    __global__ void concat_kernel_float(ConcatParams *params, float *output)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= params->outer_dims * params->output_axis_size * params->inner_dims)
        {
            return;
        }

        size_t outer_idx = idx / (params->output_axis_size * params->inner_dims);
        size_t axis_idx = (idx / params->inner_dims) % params->output_axis_size;
        size_t inner_idx = idx % params->inner_dims;

        int tensor_idx = 0;
        size_t current_offset = 0;

        for (tensor_idx = 0; tensor_idx < params->num_tensors; tensor_idx++)
        {
            size_t tensor_axis_size = params->input_axis_sizes[tensor_idx];
            if (axis_idx < current_offset + tensor_axis_size)
            {
                break;
            }
            current_offset += tensor_axis_size;
        }

        if (tensor_idx >= params->num_tensors)
        {
            return;
        }

        size_t local_axis_idx = axis_idx - current_offset;
        float *input_tensor = (float *)params->input_ptrs[tensor_idx];

        size_t input_offset = outer_idx * params->input_axis_sizes[tensor_idx] * params->inner_dims;
        input_offset += local_axis_idx * params->inner_dims;
        input_offset += inner_idx;

        size_t output_offset = outer_idx * params->output_axis_size * params->inner_dims;
        output_offset += axis_idx * params->inner_dims;
        output_offset += inner_idx;

        output[output_offset] = input_tensor[input_offset];
    }

    void get_grid_config(size_t size, int *grid_size, int *block_size)
    {
        *block_size = 256;
        *grid_size = (size + *block_size - 1) / *block_size;
    }

    int launch_scale_kernel_host(float *data, size_t size, float min_value, float max_value)
    {
        int grid_size, block_size;
        get_grid_config(size, &grid_size, &block_size);

        scale_kernel<<<grid_size, block_size>>>(data, size, min_value, max_value);

        if (cudaPeekAtLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        {
            return FAILURE;
        }

        return SUCCESS;
    }

    void launch_fill_kernel(float *data, float value, size_t size)
    {
        int threads = 256;
        int blocks = min(32, (int)((size + threads - 1) / threads));
        fill_kernel<<<blocks, threads>>>(data, value, size);
    }

    int launch_concat_kernel_host(
        tensor_t **input_tensors,
        int num_tensors,
        tensor_t *output_tensor,
        int axis,
        size_t *input_axis_offsets,
        size_t *input_strides_axis,
        size_t output_stride_axis,
        size_t outer_dims,
        size_t inner_dims,
        int output_axis_size)
    {
        size_t total_elements = outer_dims * output_axis_size * inner_dims;
        int grid_size, block_size;
        get_grid_config(total_elements, &grid_size, &block_size);

        ConcatParams params;
        params.num_tensors = num_tensors;
        params.outer_dims = outer_dims;
        params.inner_dims = inner_dims;
        params.output_stride = output_stride_axis;
        params.output_axis_size = output_axis_size;

        for (int i = 0; i < num_tensors; i++)
        {
            params.input_ptrs[i] = input_tensors[i]->data;
            params.input_axis_sizes[i] = input_tensors[i]->shape[axis];
            params.input_axis_offsets[i] = input_axis_offsets[i];
            params.input_strides_axis[i] = input_strides_axis[i];
        }

        ConcatParams *d_params;
        cudaError_t status = cudaMalloc(&d_params, sizeof(ConcatParams));
        if (status != cudaSuccess)
        {
            return FAILURE;
        }

        status = cudaMemcpy(d_params, &params, sizeof(ConcatParams), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            cudaFree(d_params);
            return FAILURE;
        }

        switch (output_tensor->dtype)
        {
        case DTYPE_FLOAT:
            concat_kernel_float<<<grid_size, block_size>>>(
                d_params,
                (float *)output_tensor->data);
            break;
        case DTYPE_INT:
            concat_kernel_int32<<<grid_size, block_size>>>(
                d_params,
                (int *)output_tensor->data);
            break;
        default:
            cudaFree(d_params);
            return FAILURE;
        }

        status = cudaPeekAtLastError();
        if (status != cudaSuccess)
        {
            cudaFree(d_params);
            return FAILURE;
        }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
        {
            cudaFree(d_params);
            return FAILURE;
        }

        cudaFree(d_params);
        return SUCCESS;
    }
    void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n)
    {
        int threads = 256;
        int blocks = min(32, (n + threads - 1) / threads);
        clip_kernel<<<blocks, threads>>>(a, min_val, max_val, result, n);
    }

    void launch_relu_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = min(32, (n + threads - 1) / threads);
        relu_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_sigmoid_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = min(32, (n + threads - 1) / threads);
        sigmoid_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_tanh_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = min(32, (n + threads - 1) / threads);
        tanh_kernel<<<blocks, threads>>>(a, result, n);
    }
}