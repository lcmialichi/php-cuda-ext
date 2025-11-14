#ifndef SCALAR_OPS_H
#define SCALAR_OPS_H

#include <stddef.h>
#include "operations.h"

#ifdef __cplusplus
extern "C" {
#endif

void launch_scalar_add_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_subtract_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_multiply_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_divide_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_power_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_greater_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_less_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_equal_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_not_equal_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_greater_equal_kernel(float *a, float scalar, float *result, size_t n);
void launch_scalar_less_equal_kernel(float *a, float scalar, float *result, size_t n);

#ifdef __cplusplus
}
#endif

using scalar_fn = void (*)(float *a, float scalar, float *result, size_t n);

typedef struct {
    int op;
    scalar_fn fn;
} ScalarDispatchEntry;

extern ScalarDispatchEntry scalar_dispatch[];

#endif
