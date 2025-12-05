#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "tensor.h"
#include <stdbool.h>

typedef enum
{
    MODEL_ELEMENT_WISE,
    MODEL_REDUCTION,
    MODEL_COMPUTE_CALL,
    MODEL_CONCAT,
    METADATA_TRANSFORM
} kernel_model_t;

typedef enum
{
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_EXP,
    OP_SQRT,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_ABS,
    OP_NEG,
    OP_SELECT,
    OP_CLAMP,
    OP_CEIL,
    OP_FLOOR,
    OP_ROUND,

    OP_GT,
    OP_LT,
    OP_EQ,
    OP_NE,
    OP_GE,
    OP_LE,

    OP_REDUCE_SUM,
    OP_REDUCE_MEAN,
    OP_REDUCE_MAX,
    OP_REDUCE_MIN,
    OP_REDUCE_PROD,

    OP_ARG_MAX,
    OP_ARG_MIN,

    OP_CONCAT,

    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_SLICE,

    OP_MATMUL
} operation_type_t;

typedef enum
{
    OP_TYPE_TENSOR_TENSOR,
    OP_TYPE_SCALAR_TENSOR,
    OP_TYPE_TENSOR_SCALAR,
    OP_TYPE_UNARY_TENSOR,
    OP_TYPE_NO_OPERAND,
} operation_arity_t;

typedef struct _operation_t
{
    operation_type_t type;
    operation_arity_t arity;
    tensor_t *result;
    kernel_model_t model;
    union
    {
        struct
        {
            tensor_t *a;
            tensor_t *b;
        } tensor_tensor;

        struct
        {
            tensor_t *tensor;
            float scalar;
        } tensor_scalar;

        struct
        {
            float scalar;
            tensor_t *tensor;
        } scalar_tensor;

        struct
        {
            tensor_t *tensor;
        } unary;

        struct
        {
            double a;
            double b;
        } scalar_scalar;
    } operands;

    int output_ndims;
    int output_shape[MAX_DIMS];
    int output_id;
    char output_alias[8];

    union
    {
        struct
        {
            int axis;
            bool keep_dims;
        } reduction;

        struct
        {
            int new_shape[MAX_DIMS];
        } reshape;

        struct
        {
            int perm[MAX_DIMS];
        } transpose;

        struct
        {
            int slice_starts[MAX_DIMS];
            int slice_ends[MAX_DIMS];
            int slice_steps[MAX_DIMS];
        } slice;

    } params;

} operation_t;

typedef struct _op_list_node_t
{
    operation_t *op;
    struct _op_list_node_t *next;
} op_list_node_t;

typedef struct _op_list_t
{
    op_list_node_t *head;
    op_list_node_t *tail;
    size_t count;
} op_list_t;

typedef struct _fusion_context_t
{
    op_list_t operation_nodes;
    tensor_t *trace_output;
    struct
    {
        bool is_active;
        int next_id;
        int temp_id_count;
        int input_id_count;
        int constant_id_count;
        int op_counter;
    } tracker;
} fusion_context_t;

tensor_t *create_tensor_operation_node(operation_type_t op_type, tensor_t *a, tensor_t *b);
tensor_t *create_scalar_operation_node(operation_type_t op_type, tensor_t *a, float b);
tensor_t *create_unary_operation_node(operation_type_t op_type, tensor_t *a);

int prepare_broadcast_operation(tensor_t *a, tensor_t *b,
                                int *result_shape, int *result_dims,
                                int *a_strides, int *b_strides,
                                size_t *total_elements);

int calculate_reduction_shape(tensor_t *input, int axis, int *result_shape, size_t *total_elements_out_ptr);
int prepare_matmul_result_shape(int a_ndims, int *a_shape, int b_ndims, int *b_shape, int *result_ndims, int *result_shape);
#endif