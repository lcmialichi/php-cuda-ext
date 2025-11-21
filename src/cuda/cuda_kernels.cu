#include <cuda_runtime.h>
#include <math.h>
#include "cuda_kernels.h"
#include <float.h>
#include <cstdio>

extern "C"
{

    __global__ void scatter_write_kernel(
        float *dest_data,
        const float *src_data,
        const int *indices,
        size_t num_indices,
        size_t slice_size)
    {
        int index_idx = blockIdx.x;
        if (index_idx >= num_indices)
            return;

        int dest_dim_index = indices[index_idx];

        size_t dest_offset = dest_dim_index * slice_size;

        size_t src_offset = index_idx * slice_size;
        size_t local_idx = threadIdx.x;

        while (local_idx < slice_size)
        {
            size_t global_dest_pos = dest_offset + local_idx;
            size_t global_src_pos = src_offset + local_idx;

            dest_data[global_dest_pos] = src_data[global_src_pos];

            local_idx += blockDim.x;
        }
    }

    __global__ void fill_kernel(float *data, float value, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            data[idx] = value;
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

    __global__ void scatter_copy_kernel(
        float *dest_data,
        const float *src_data,
        const int *indices,
        size_t num_indices,
        size_t slice_size)
    {
        int index_idx = blockIdx.x;
        if (index_idx >= num_indices)
            return;

        int dest_dim_index = indices[index_idx];
        size_t dest_offset = dest_dim_index * slice_size;
        size_t src_offset = index_idx * slice_size;
        size_t local_idx = threadIdx.x;

        while (local_idx < slice_size)
        {
            size_t global_dest_pos = dest_offset + local_idx;
            size_t global_src_pos = src_offset + local_idx;

            dest_data[global_dest_pos] = src_data[global_src_pos];

            local_idx += blockDim.x;
        }
    }

    void launch_scatter(float *dest_data,
                        const float *src_data,
                        const int *indices,
                        size_t num_indices,
                        size_t slice_size)
    {
        int threads = 256;
        if (threads > slice_size)
        {
            threads = (int)slice_size;
        }

        int blocks = (int)num_indices;

        scatter_copy_kernel<<<blocks, threads>>>(dest_data, src_data, indices, num_indices, slice_size);
    }

    void launch_fill_kernel(float *data, float value, size_t size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_kernel<<<blocks, threads>>>(data, value, size);
    }

    void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        clip_kernel<<<blocks, threads>>>(a, min_val, max_val, result, n);
    }

    void launch_relu_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_sigmoid_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sigmoid_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_tanh_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        tanh_kernel<<<blocks, threads>>>(a, result, n);
    }
}