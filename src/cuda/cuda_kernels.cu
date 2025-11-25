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

    void launch_fill_kernel(float *data, float value, size_t size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_kernel<<<blocks, threads>>>(data, value, size);
    }

    void get_grid_config(size_t size, int *grid_size, int *block_size)
    {
        const int max_threads = 512;
        *block_size = (size < max_threads) ? (int)size : max_threads;
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