#include <cuda_runtime.h>
#include <math.h>
#include "cuda_kernels.h"
#include <float.h>
#include <cstdio>

extern "C"
{

    __global__ void sqrt_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = sqrtf(a[i]);
    }

    __global__ void exp_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = expf(a[i]);
    }

    __global__ void log_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = logf(a[i]);
    }

    __global__ void sin_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = sinf(a[i]);
    }

    __global__ void cos_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = cosf(a[i]);
    }

    __global__ void tan_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = tanf(a[i]);
    }

    __global__ void abs_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = fabsf(a[i]);
    }

    __global__ void negate_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = -a[i];
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

    void launch_sqrt_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sqrt_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_exp_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        exp_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_log_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        log_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_sin_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sin_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_cos_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        cos_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_tan_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        tan_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_abs_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        abs_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_negate_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        negate_kernel<<<blocks, threads>>>(a, result, n);
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