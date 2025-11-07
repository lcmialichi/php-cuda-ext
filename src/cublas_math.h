#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif
    int cuda_wrapper_init_blas();
    void cuda_wrapper_shutdown_blas();
    int cuda_wrapper_matrix_create(int rows, int cols, float **matrix);
    void cuda_wrapper_matrix_free(float *matrix);
    int cuda_wrapper_matrix_multiply(const float *A, const float *B, float *C,
                                     int m, int n, int k)
    int cuda_wrapper_matrix_add(const float *A, const float *B, float *C,
                                int rows, int cols)
    int cuda_wrapper_matrix_divide(const float *A, const float *B, float *C,
                                   int rows, int cols)

#ifdef __cplusplus
}
#endif

#endif