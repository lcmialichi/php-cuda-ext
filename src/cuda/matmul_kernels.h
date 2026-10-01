#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_matmul_launcher(float *a, float *b, float *c,
                        int m, int n, int k,
                        size_t a_stride0, size_t a_stride1,
                        size_t b_stride0, size_t b_stride1,
                        size_t c_stride0, size_t c_stride1);

int cuda_batched_matmul_nd_launcher(
    float *a, float *b, float *c,
    int *shape_a, size_t *stride_a, int nd_a,
    int *shape_b, size_t *stride_b, int nd_b,
    int *shape_c, size_t *stride_c, int nd_c);

#ifdef __cplusplus
}
#endif

#endif