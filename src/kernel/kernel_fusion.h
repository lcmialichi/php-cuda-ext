#ifndef KERNEL_FUSION_H
#define KERNEL_FUSION_H

#include "tensor.h"
#include "operations.h"

void start_kernel_fusions();
tensor_t *stop_kernel_fusions();
bool is_tracing();
void set_current_trace_output(tensor_t *t);
void op_list_add(op_list_t *list, operation_t *new_op);
void op_list_print(op_list_t *list);

#endif