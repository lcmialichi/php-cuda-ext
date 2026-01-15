#ifndef UNARY_OPS_H
#define UNARY_OPS_H

#include <stddef.h>
#include "../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_unary_exp_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_sqrt_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_log_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_sin_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_cos_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_tan_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_abs_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_neg_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_floor_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_ceil_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_unary_round_kernel(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);

#ifdef __cplusplus
}
#endif

typedef void (*unary_fn)(float *base, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);

typedef struct
{
    int op;
    unary_fn fn;
} UnaryDispatchEntry;

extern UnaryDispatchEntry unary_dispatch[];

#endif
