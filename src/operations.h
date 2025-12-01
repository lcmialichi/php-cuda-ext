#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "tensor.h"

typedef enum
{
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_GT,
    OP_LT,
    OP_EQ,
    OP_NE,
    OP_GE,
    OP_LE,
    OP_EXP,
    OP_SQRT,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_ABS,
    OP_NEG,
    OP_REDUCE_SUM,
    OP_REDUCE_MEAN,
    OP_REDUCE_MAX,
    OP_REDUCE_MIN,
    OP_REDUCE_PROD,
    OP_ARG_MAX,
    OP_ARG_MIN
} operation_type_t;

int prepare_broadcast_operation(tensor_t *a, tensor_t *b,
                                int *result_shape, int *result_dims,
                                int *a_strides, int *b_strides,
                                size_t *total_elements);

int calculate_reduction_shape(tensor_t *input, int axis, int *result_shape, size_t *total_elements_out_ptr);
int prepare_matmul_result_shape(int a_ndims, int *a_shape, int b_ndims, int *b_shape, int *result_ndims, int *result_shape);
#endif