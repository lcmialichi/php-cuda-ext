#include "trace_ops.h"
#include "operations.h"
#include "tensor.h"
#include "kernel_fusion.h"
#include "php.h"

tensor_t *trace_binary_operation(operation_type_t op_type, tensor_t *a, tensor_t *b)
{
    tensor_t *result_proxy = create_tensor_operation_node(op_type, a, b);

    if (result_proxy == NULL)
    {
        zend_throw_error(NULL, "Failed to allocate tensor proxy for tracing.");
        return NULL;
    }

    set_current_trace_output(result_proxy);

    return result_proxy;
}

tensor_t *trace_scalar_operation(operation_type_t op_type, tensor_t *a, float b)
{
    tensor_t *result_proxy = create_scalar_operation_node(op_type, a, b);

    if (result_proxy == NULL)
    {
        zend_throw_error(NULL, "Failed to allocate tensor proxy for tracing.");
        return NULL;
    }

    set_current_trace_output(result_proxy);

    return result_proxy;
}
