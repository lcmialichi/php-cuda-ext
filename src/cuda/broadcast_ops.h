#ifndef BROADCAST_OPS_H
#define BROADCAST_OPS_H

#include "../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void (*broadcast_fn)(float *a, float *b, float *result,
                                 int *a_strides, int a_dims,
                                 int *b_strides, int b_dims,
                                 int *result_shape, int result_dims,
                                 size_t total_elements, size_t a_base_offset,
                                 size_t b_base_offset);
    typedef struct
    {
        operation_type_t op;
        broadcast_fn fn;
    } BroadcastDispatchEntry;

    void launch_broadcast(
        void *a, void *b, void *result,
        dtype_t dtype, operation_type_t op_type,
        int *a_strides, int a_dims,
        int *b_strides, int b_dims,
        int *result_shape, int result_dims,
        size_t total_elements, size_t a_offset, size_t b_offset);

    void launch_broadcast_add(float *a, float *b, float *result,
                              int *a_strides, int a_dims,
                              int *b_strides, int b_dims,
                              int *result_shape, int result_dims,
                              size_t total_elements, size_t a_base_offset,
                              size_t b_base_offset);

    void launch_broadcast_subtract(float *a, float *b, float *result,
                                   int *a_strides, int a_dims,
                                   int *b_strides, int b_dims,
                                   int *result_shape, int result_dims,
                                   size_t total_elements, size_t a_base_offset,
                                   size_t b_base_offset);

    void launch_broadcast_multiply(float *a, float *b, float *result,
                                   int *a_strides, int a_dims,
                                   int *b_strides, int b_dims,
                                   int *result_shape, int result_dims,
                                   size_t total_elements, size_t a_base_offset,
                                   size_t b_base_offset);

    void launch_broadcast_divide(float *a, float *b, float *result,
                                 int *a_strides, int a_dims,
                                 int *b_strides, int b_dims,
                                 int *result_shape, int result_dims,
                                 size_t total_elements, size_t a_base_offset,
                                 size_t b_base_offset);

    void launch_broadcast_power(float *a, float *b, float *result,
                                int *a_strides, int a_dims,
                                int *b_strides, int b_dims,
                                int *result_shape, int result_dims,
                                size_t total_elements, size_t a_base_offset,
                                size_t b_base_offset);

    void launch_broadcast_greater(float *a, float *b, float *result,
                                  int *a_strides, int a_dims,
                                  int *b_strides, int b_dims,
                                  int *result_shape, int result_dims,
                                  size_t total_elements, size_t a_base_offset,
                                  size_t b_base_offset);

    void launch_broadcast_less(float *a, float *b, float *result,
                               int *a_strides, int a_dims,
                               int *b_strides, int b_dims,
                               int *result_shape, int result_dims,
                               size_t total_elements, size_t a_base_offset,
                               size_t b_base_offset);

    void launch_broadcast_equal(float *a, float *b, float *result,
                                int *a_strides, int a_dims,
                                int *b_strides, int b_dims,
                                int *result_shape, int result_dims,
                                size_t total_elements, size_t a_base_offset,
                                size_t b_base_offset);

    void launch_broadcast_not_equal(float *a, float *b, float *result,
                                    int *a_strides, int a_dims,
                                    int *b_strides, int b_dims,
                                    int *result_shape, int result_dims,
                                    size_t total_elements, size_t a_base_offset,
                                    size_t b_base_offset);

    void launch_broadcast_greater_equal(float *a, float *b, float *result,
                                        int *a_strides, int a_dims,
                                        int *b_strides, int b_dims,
                                        int *result_shape, int result_dims,
                                        size_t total_elements, size_t a_base_offset,
                                        size_t b_base_offset);

    void launch_broadcast_less_equal(float *a, float *b, float *result,
                                     int *a_strides, int a_dims,
                                     int *b_strides, int b_dims,
                                     int *result_shape, int result_dims,
                                     size_t total_elements, size_t a_base_offset,
                                     size_t b_base_offset);

#ifdef __cplusplus
}
#endif

#endif