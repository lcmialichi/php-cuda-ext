#ifndef SCALAR_OPS_H
#define SCALAR_OPS_H

#include <stddef.h>
#include "../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_scalar_add_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_subtract_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_multiply_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_divide_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_power_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_greater_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_less_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_equal_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_not_equal_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_greater_equal_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);
    void launch_scalar_less_equal_kernel(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);

#ifdef __cplusplus
}
#endif

typedef void (*scalar_fn)(float *base, float scalar, float *result, size_t base_offset, int *shape, size_t *strides, int ndims, size_t total_size);

typedef struct
{
    int op;
    scalar_fn fn;
} ScalarDispatchEntry;

extern ScalarDispatchEntry scalar_dispatch[];

#endif
