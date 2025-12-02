#ifndef KERNEL_FUSION_H
#define KERNEL_FUSION_H

#include "tensor.h"
#include "operations.h"

void start_kernel_fusions();
tensor_t *stop_kernel_fusions();
bool is_tracing();
void set_current_trace_output(tensor_t *t);
void op_list_add(op_list_t *list, operation_t *new_op);
void op_list_print();

operation_t *fusion_create_tensor_tensor_op(operation_type_t type, 
                                           tensor_t *a, tensor_t *b,
                                           tensor_t *result);
                                           
operation_t *fusion_create_tensor_scalar_op(operation_type_t type,
                                           tensor_t *tensor, float scalar,
                                           tensor_t *result);

operation_t *fusion_create_scalar_tensor_op(operation_type_t type,
                                           float scalar, tensor_t *tensor,
                                           tensor_t *result);


void fusion_link_tensor(tensor_t *tensor, operation_t *op);
void fusion_tag_as_constant(tensor_t *tensor, const char *const_type);

#endif