#include "trace_ops.h"
#include "operations.h"
#include "tensor.h"
#include "kernel_fusion.h"
#include "php.h"

tensor_t *trace_binary_operation(operation_type_t op_type, tensor_t *a, tensor_t *b)
{
    int result_shape[MAX_DIMS];
    int result_dims;

    operation_t *op_node = create_tensor_operation_node(op_type, a, b);
    
    if (op_node == NULL) {
        zend_throw_error(NULL, "Failed to allocate operation node for tracing.");
        return NULL;
    }

    tensor_t *result_proxy = create_new_tensor_proxy(result_shape, result_dims, op_node);
    
    if (result_proxy == NULL) {
        zend_throw_error(NULL, "Failed to allocate tensor proxy for tracing.");
        return NULL;
    }

    set_current_trace_output(result_proxy);

    return result_proxy;
}
