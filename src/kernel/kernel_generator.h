#ifndef KERNEL_GENERATOR_H
#define KERNEL_GENERATOR_H

#include "operations.h"

typedef struct
{
    tensor_t **tensors;
    int count;
    int capacity;
} tensor_list_t;

#define TMP_PREFIX "tmp_"
#define INP_PREFIX "inp_"
#define OUT_PREFIX "out_"

typedef struct _kernel_generator
{
    int id;
    kernel_model_t kernel_type;

    fusion_context_t *context;
    op_list_t required_ops;

    int block_size;
    int grid_size;
    tensor_t *final_output;

    tensor_list_t inputs;
    tensor_list_t outputs;
    tensor_list_t temps;

    char *header_code;
    char *device_code;
    char *kernel_code;
    char *launch_code;

    int total_threads;
    size_t memory_bytes;
    int num_params;

    struct _kernel_generator **dependencies;
    int num_dependencies;
    int num_ops;
    int analyzed;
    int generated;

} kernel_generator_t;

typedef struct _multi_kernel_generator
{
    kernel_generator_t **kernels;
    int kernel_count;
    int kernel_capacity;
    tensor_t *final_output;

    tensor_list_t inter_kernel_tensors;
    int default_block_size;

    char *header_code;
    char *device_code;
    char *combined_kernel_code;
    char *combined_launch_code;

    int analyzed;
    int generated;

} multi_kernel_generator;

#define KERNEL_OPLIST_FOREACH(current, op_var)           \
    for (op_list_node_t *_node_ = (current);             \
         _node_ != NULL && ((op_var = _node_->op) || 1); \
         _node_ = _node_->next)                          \
        if ((op_var) != NULL)

#define MULTI_KERNEL_FOREACH(multi, kg_var) \
    for (int __i = 0; __i < (multi)->kernel_count && ((kg_var) = (multi)->kernels[__i]); __i++)

#define SET_TENSOR_ACCESS(TARGET, TENSOR)                                                     \
    if ((TENSOR)->trace.tensor_type == TENSOR_TYPE_TEMP)                                      \
    {                                                                                         \
        snprintf(TARGET, sizeof(TARGET), "%s%s", TMP_PREFIX, get_operand_alias(TENSOR));      \
    }                                                                                         \
    else if ((TENSOR)->trace.tensor_type == TENSOR_TYPE_INPUT)                                \
    {                                                                                         \
        snprintf(TARGET, sizeof(TARGET), "%s%s[idx]", INP_PREFIX, get_operand_alias(TENSOR)); \
    }                                                                                         \
    else if ((TENSOR)->trace.tensor_type == TENSOR_TYPE_OUTPUT)                               \
    {                                                                                         \
        snprintf(TARGET, sizeof(TARGET), "%s%s[idx]", OUT_PREFIX, get_operand_alias(TENSOR)); \
    }

#define SET_RESULT_ACCESS(TARGET, TENSOR)                                                     \
    if ((TENSOR)->trace.tensor_type == TENSOR_TYPE_OUTPUT)                                    \
    {                                                                                         \
        snprintf(TARGET, sizeof(TARGET), "%s%s[idx]", OUT_PREFIX, get_operand_alias(TENSOR)); \
    }                                                                                         \
    else if ((TENSOR)->trace.tensor_type == TENSOR_TYPE_TEMP)                                 \
    {                                                                                         \
        snprintf(TARGET, sizeof(TARGET), "%s%s", TMP_PREFIX, get_operand_alias(TENSOR));   \
    }                                                                                         \
    else if ((TENSOR)->trace.tensor_type == TENSOR_TYPE_INPUT)                                \
    {                                                                                         \
        snprintf(TARGET, sizeof(TARGET), "%s%s", INP_PREFIX, get_result_alias(TENSOR));       \
    }

multi_kernel_generator *multi_kernel_create(tensor_t *final_output);
void multi_kernel_destroy(multi_kernel_generator *multi);
bool multi_kernel_analyze_and_split(multi_kernel_generator *multi);
bool multi_kernel_generate(multi_kernel_generator *multi);
void multi_kernel_generator_print(multi_kernel_generator *multi);

char *mk_get_code_as_c(multi_kernel_generator *multi);

// bool kernel_generator_compile(kernel_generator_t *gen);
// bool kernel_generator_execute(kernel_generator_t *gen);

#endif