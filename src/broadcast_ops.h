#ifndef BROADCAST_OPS_H
#define BROADCAST_OPS_H

#include "operations.h"

#ifdef __cplusplus
extern "C"
{
#endif

    using broadcast_fn = void (*)(float *a, float *b, float *result,
                                  int *a_strides, int a_dims,
                                  int *b_strides, int b_dims,
                                  int *result_shape, int result_dims,
                                  size_t total_elements);
    typedef struct
    {
        int op;
        broadcast_fn fn;
    } BroadcastDispatchEntry;

    void launch_broadcast_add(float *a, float *b, float *result,
                              int *a_strides, int a_dims,
                              int *b_strides, int b_dims,
                              int *result_shape, int result_dims,
                              size_t total_elements);

    void launch_broadcast_subtract(float *a, float *b, float *result,
                                   int *a_strides, int a_dims,
                                   int *b_strides, int b_dims,
                                   int *result_shape, int result_dims,
                                   size_t total_elements);

    void launch_broadcast_multiply(float *a, float *b, float *result,
                                   int *a_strides, int a_dims,
                                   int *b_strides, int b_dims,
                                   int *result_shape, int result_dims,
                                   size_t total_elements);

    void launch_broadcast_divide(float *a, float *b, float *result,
                                 int *a_strides, int a_dims,
                                 int *b_strides, int b_dims,
                                 int *result_shape, int result_dims,
                                 size_t total_elements);

    void launch_broadcast_power(float *a, float *b, float *result,
                                int *a_strides, int a_dims,
                                int *b_strides, int b_dims,
                                int *result_shape, int result_dims,
                                size_t total_elements);

    void launch_broadcast_greater(float *a, float *b, float *result,
                                  int *a_strides, int a_dims,
                                  int *b_strides, int b_dims,
                                  int *result_shape, int result_dims,
                                  size_t total_elements);

    void launch_broadcast_less(float *a, float *b, float *result,
                               int *a_strides, int a_dims,
                               int *b_strides, int b_dims,
                               int *result_shape, int result_dims,
                               size_t total_elements);

    void launch_broadcast_equal(float *a, float *b, float *result,
                                int *a_strides, int a_dims,
                                int *b_strides, int b_dims,
                                int *result_shape, int result_dims,
                                size_t total_elements);

    void launch_broadcast_not_equal(float *a, float *b, float *result,
                                    int *a_strides, int a_dims,
                                    int *b_strides, int b_dims,
                                    int *result_shape, int result_dims,
                                    size_t total_elements);

    void launch_broadcast_greater_equal(float *a, float *b, float *result,
                                        int *a_strides, int a_dims,
                                        int *b_strides, int b_dims,
                                        int *result_shape, int result_dims,
                                        size_t total_elements);

    void launch_broadcast_less_equal(float *a, float *b, float *result,
                                     int *a_strides, int a_dims,
                                     int *b_strides, int b_dims,
                                     int *result_shape, int result_dims,
                                     size_t total_elements);

#ifdef __cplusplus
}
#endif

#endif