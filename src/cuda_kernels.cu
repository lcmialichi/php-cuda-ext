#include <cuda_runtime.h>
#include "cuda_kernels.h" 

extern "C" {
    __global__ void divide_kernel(float *a, float *b, float *result, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            result[i] = a[i] / b[i];
        }
    }
    
    __global__ void scalar_multiply_kernel(float *a, float scalar, float *result, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            result[i] = a[i] * scalar;
        }
    }
    
    void launch_divide_kernel(float *a, float *b, float *result, int n) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        divide_kernel<<<blocks, threads>>>(a, b, result, n);
    }
    
    void launch_scalar_multiply_kernel(float *a, float scalar, float *result, int n) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scalar_multiply_kernel<<<blocks, threads>>>(a, scalar, result, n);
    }
}