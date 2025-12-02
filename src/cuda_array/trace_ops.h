#ifndef TRACE_OPS_H
#define TRACE_OPS_H

#include "tensor.h"
#include "operations.h"

tensor_t *trace_binary_operation(operation_type_t op_type, tensor_t *a, tensor_t *b);
tensor_t *trace_scalar_operation(operation_type_t op_type, tensor_t *a, float b);

#endif
