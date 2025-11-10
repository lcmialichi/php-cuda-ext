#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Divisão elemento a elemento
void launch_divide_kernel(float *a, float *b, float *result, int n);

// Multiplicação por escalar  
void launch_scalar_multiply_kernel(float *a, float scalar, float *result, int n);

#ifdef __cplusplus
}
#endif

#endif