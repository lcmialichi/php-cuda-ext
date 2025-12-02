#include "php.h"

#ifdef ZTS
#include "TSRM.h"
#endif

#include "cuda_globals.h"
#include "tensor.h"
#include "operations.h"
#include <string.h>
#include "php.h"
#include "php.h"
#include "cuda_globals.h"
#include "tensor.h"
#include "operations.h"
#include <string.h>

static void fusion_auto_tag_tensor(tensor_t *tensor);
static operation_t *fusion_create_base_op(operation_type_t type, tensor_t *result);

void op_list_init(op_list_t *list)
{
    list->head = NULL;
    list->tail = NULL;
    list->count = 0;
}

void start_kernel_fusions()
{
    fusion_context_t *context = (fusion_context_t *)emalloc(sizeof(fusion_context_t));
    if (!context)
    {
        zend_throw_error(NULL, "Failed to allocate fusion context memory.");
        return;
    }

    memset(context, 0, sizeof(fusion_context_t));
    op_list_init(&context->operation_nodes);

    context->tracker.is_active = true;
    context->tracker.next_temp_id = 0;
    context->tracker.next_input_id = 0;
    context->tracker.next_constant_id = 0;
    context->tracker.op_counter = 0;

    CUDA_G(current_fusion_context) = context;
    CUDA_G(is_tracing_enabled) = true;
}

void fusion_link_tensor(tensor_t *tensor, operation_t *op)
{
    if (!tensor || !op)
        return;

    tensor->trace.defining_op = op;
    tensor->trace.expr_id = op->output_id;
    snprintf(tensor->trace.expr_alias, sizeof(tensor->trace.expr_alias),
             "%s", op->output_alias);
}

void op_list_add(op_list_t *list, operation_t *new_op)
{
    op_list_node_t *node = (op_list_node_t *)emalloc(sizeof(op_list_node_t));
    if (!node)
    {
        zend_throw_error(NULL, "Failed to allocate op_list_node memory.");
        return;
    }

    memset(node, 0, sizeof(op_list_node_t));
    node->op = new_op;
    node->next = NULL;

    if (list->tail == NULL)
    {
        list->head = node;
    }
    else
    {
        list->tail->next = node;
    }
    list->tail = node;
    list->count++;
}

void fusion_tag_as_constant(tensor_t *tensor, const char *const_type)
{
    if (!tensor)
        return;

    fusion_context_t *context = CUDA_G(current_fusion_context);
    if (!context || !context->tracker.is_active)
        return;

    if (tensor->trace.expr_alias[0] != '\0')
        return;

    if (const_type && const_type[0])
    {
        snprintf(tensor->trace.expr_alias, sizeof(tensor->trace.expr_alias),
                 "%s%d", const_type, context->tracker.next_constant_id++);
    }
    else
    {
        snprintf(tensor->trace.expr_alias, sizeof(tensor->trace.expr_alias),
                 "C%d", context->tracker.next_constant_id++);
    }

    tensor->trace.expr_id = -2;
    tensor->trace.defining_op = NULL;
}

operation_t *fusion_create_tensor_tensor_op(operation_type_t type,
                                            tensor_t *a, tensor_t *b,
                                            tensor_t *result)
{
    operation_t *op = fusion_create_base_op(type, result);
    if (!op)
        return NULL;

    op->arity = OP_TYPE_TENSOR_TENSOR;
    op->operands.tensor_tensor.a = a;
    op->operands.tensor_tensor.b = b;

    if (a)
        TENSOR_ADD_REF(a);
    if (b)
        TENSOR_ADD_REF(b);

    return op;
}

operation_t *fusion_create_tensor_scalar_op(operation_type_t type,
                                            tensor_t *tensor, float scalar,
                                            tensor_t *result)
{
    operation_t *op = fusion_create_base_op(type, result);
    if (!op)
        return NULL;

    op->arity = OP_TYPE_TENSOR_SCALAR;
    op->operands.tensor_scalar.tensor = tensor;
    op->operands.tensor_scalar.scalar = scalar;

    if (tensor)
        TENSOR_ADD_REF(tensor);

    return op;
}

operation_t *fusion_create_scalar_tensor_op(operation_type_t type,
                                            float scalar, tensor_t *tensor,
                                            tensor_t *result)
{
    operation_t *op = fusion_create_base_op(type, result);
    if (!op)
        return NULL;

    op->arity = OP_TYPE_SCALAR_TENSOR;
    op->operands.scalar_tensor.scalar = scalar;
    op->operands.scalar_tensor.tensor = tensor;

    if (tensor)
        TENSOR_ADD_REF(tensor);

    return op;
}

operation_t *fusion_create_unary_op(operation_type_t type,
                                    tensor_t *tensor,
                                    tensor_t *result)
{
    operation_t *op = fusion_create_base_op(type, result);
    if (!op)
        return NULL;

    op->arity = OP_TYPE_UNARY_TENSOR;
    op->operands.unary.tensor = tensor;

    if (tensor)
        TENSOR_ADD_REF(tensor);

    return op;
}

void fusion_tag_input_tensor(tensor_t *tensor, const char *alias_prefix, int index)
{
    if (!tensor)
        return;

    fusion_context_t *context = CUDA_G(current_fusion_context);
    if (!context || !context->tracker.is_active)
        return;

    snprintf(tensor->trace.expr_alias, sizeof(tensor->trace.expr_alias),
             "%s%d", alias_prefix, index);
    tensor->trace.expr_id = -1;
    tensor->trace.defining_op = NULL;
}

const char *fusion_get_tensor_alias(const tensor_t *tensor)
{
    if (!tensor || tensor->trace.expr_alias[0] == '\0')
    {
        return "?";
    }
    return tensor->trace.expr_alias;
}

tensor_t *stop_kernel_fusions()
{
    CUDA_G(is_tracing_enabled) = false;

    fusion_context_t *context = CUDA_G(current_fusion_context);
    tensor_t *result = NULL;

    if (context != NULL)
    {
        result = context->trace_output;

        op_list_node_t *current = context->operation_nodes.head;
        while (current != NULL)
        {
            op_list_node_t *next = current->next;

            if (current->op != NULL)
            {
                efree(current->op);
            }

            efree(current);
            current = next;
        }

        context->operation_nodes.head = NULL;
        context->operation_nodes.tail = NULL;
        context->operation_nodes.count = 0;

        efree(context);
        CUDA_G(current_fusion_context) = NULL;
    }

    return result;
}

bool is_tracing()
{
    return CUDA_G(is_tracing_enabled);
}

void set_current_trace_output(tensor_t *t)
{
    fusion_context_t *context = CUDA_G(current_fusion_context);

    if (context != NULL)
    {
        context->trace_output = t;
        if (t && context->operation_nodes.tail && context->operation_nodes.tail->op)
        {
            operation_t *last_op = context->operation_nodes.tail->op;
            fusion_link_tensor(t, last_op);
        }
    }
    else
    {
        php_error_docref(NULL, E_WARNING,
                         "set_current_trace_output called when tracing is inactive (TLS context is NULL).");
    }
}

static const char *operation_type_to_string(operation_type_t type)
{
    switch (type)
    {
    case OP_ADD:
        return "+";
    case OP_SUB:
        return "-";
    case OP_MUL:
        return "*";
    case OP_DIV:
        return "/";
    case OP_POW:
        return "**";
    case OP_GT:
        return "GT";
    case OP_LT:
        return "LT";
    case OP_EQ:
        return "EQ";
    case OP_NE:
        return "NE";
    case OP_GE:
        return "GE";
    case OP_LE:
        return "LE";
    case OP_EXP:
        return "EXP";
    case OP_SQRT:
        return "SQRT";
    case OP_LOG:
        return "LOG";
    case OP_SIN:
        return "SIN";
    case OP_COS:
        return "COS";
    case OP_TAN:
        return "TAN";
    case OP_ABS:
        return "ABS";
    case OP_NEG:
        return "NEG";
    case OP_REDUCE_SUM:
        return "REDUCE_SUM";
    case OP_REDUCE_MEAN:
        return "REDUCE_MEAN";
    case OP_REDUCE_MAX:
        return "REDUCE_MAX";
    case OP_REDUCE_MIN:
        return "REDUCE_MIN";
    case OP_REDUCE_PROD:
        return "REDUCE_PROD";
    case OP_ARG_MAX:
        return "ARG_MAX";
    case OP_ARG_MIN:
        return "ARG_ARG_MIN";
    default:
        return "UNKNOWN_OP";
    }
}
void op_list_print()
{
    fusion_context_t *context = CUDA_G(current_fusion_context);
    if (!context || context->operation_nodes.count == 0)
    {
        php_printf("\n--- No fusion trace active ---\n");
        return;
    }

    php_printf("\n=== Kernel Fusion Trace (%zu operations) ===\n",
               context->operation_nodes.count);

    op_list_node_t *current = context->operation_nodes.head;
    int index = 0;

    while (current != NULL)
    {
        operation_t *op = current->op;
        if (op == NULL)
        {
            current = current->next;
            index++;
            continue;
        }

        const char *op_name = operation_type_to_string(op->type);
        php_printf("[%d] %s = ", index, op->output_alias);

        switch (op->arity)
        {
        case OP_TYPE_TENSOR_TENSOR:
        {
            const char *a_alias = op->operands.tensor_tensor.a ? fusion_get_tensor_alias(op->operands.tensor_tensor.a) : "?";
            const char *b_alias = op->operands.tensor_tensor.b ? fusion_get_tensor_alias(op->operands.tensor_tensor.b) : "?";

            if (op->operands.tensor_tensor.a == op->operands.tensor_tensor.b)
            {
                php_printf("(%s %s %s)", a_alias, op_name, a_alias);
            }
            else
            {
                php_printf("(%s %s, %s)", a_alias, op_name, b_alias);
            }
            break;
        }

        case OP_TYPE_TENSOR_SCALAR:
        {
            const char *tensor_alias = op->operands.tensor_scalar.tensor ? fusion_get_tensor_alias(op->operands.tensor_scalar.tensor) : "?";
            php_printf("(%s %s, %4f)", tensor_alias, op_name, op->operands.tensor_scalar.scalar);
            break;
        }

        case OP_TYPE_SCALAR_TENSOR:
        {
            const char *tensor_alias = op->operands.scalar_tensor.tensor ? fusion_get_tensor_alias(op->operands.scalar_tensor.tensor) : "?";

            if (op->type == OP_SUB || op->type == OP_DIV || op->type == OP_POW)
            {
                const char *reverse_op =
                    (op->type == OP_SUB) ? "RSUB" : (op->type == OP_DIV) ? "RDIV"
                                                                         : "RPOW";
                php_printf("%s(%.4f, %s)", reverse_op, op->operands.scalar_tensor.scalar, tensor_alias);
            }
            else
            {
                php_printf("%s(%.4f, %s)", op_name, op->operands.scalar_tensor.scalar, tensor_alias);
            }
            break;
        }

        case OP_TYPE_UNARY_TENSOR:
        {
            const char *tensor_alias = op->operands.unary.tensor ? fusion_get_tensor_alias(op->operands.unary.tensor) : "?";
            php_printf("%s(%s)", op_name, tensor_alias);
            break;
        }

        case OP_TYPE_NO_OPERAND:
        {
            php_printf("%s()", op_name);
            break;
        }

        default:
            php_printf("UNKNOWN_ARITY");
            break;
        }

        if (op->output_ndims > 0)
        {
            php_printf("  [Shape: (");
            for (int i = 0; i < op->output_ndims; i++)
            {
                php_printf("%d%s", op->output_shape[i],
                           (i == op->output_ndims - 1) ? "" : ", ");
            }
            php_printf(")]");

            int dtype = 0;
            switch (op->arity)
            {
            case OP_TYPE_TENSOR_TENSOR:
                if (op->operands.tensor_tensor.a)
                    dtype = op->operands.tensor_tensor.a->dtype;
                break;
            case OP_TYPE_TENSOR_SCALAR:
                if (op->operands.tensor_scalar.tensor)
                    dtype = op->operands.tensor_scalar.tensor->dtype;
                break;
            case OP_TYPE_SCALAR_TENSOR:
                if (op->operands.scalar_tensor.tensor)
                    dtype = op->operands.scalar_tensor.tensor->dtype;
                break;
            case OP_TYPE_UNARY_TENSOR:
                if (op->operands.unary.tensor)
                    dtype = op->operands.unary.tensor->dtype;
                break;
            }

            if (dtype != 0)
            {
                const char *dtype_str = "Unknown";
                switch (dtype)
                {
                case 1:
                    dtype_str = "float32";
                    break;
                case 2:
                    dtype_str = "float64";
                    break;
                case 3:
                    dtype_str = "int32";
                    break;
                case 4:
                    dtype_str = "int64";
                    break;
                }
                php_printf(" [%s]", dtype_str);
            }
        }

        switch (op->type)
        {
        case OP_REDUCE_SUM:
        case OP_REDUCE_MEAN:
        case OP_REDUCE_MAX:
        case OP_REDUCE_MIN:
        case OP_REDUCE_PROD:
            php_printf(" [axis=%d, keep_dims=%s]",
                       op->params.reduction.axis,
                       op->params.reduction.keep_dims ? "true" : "false");
            break;

        case OP_RESHAPE:
            if (op->params.reshape.new_shape[0] != 0)
            {
                php_printf(" -> (");
                for (int i = 0; i < MAX_DIMS && op->params.reshape.new_shape[i] != 0; i++)
                {
                    php_printf("%d%s", op->params.reshape.new_shape[i],
                               (i < MAX_DIMS - 1 && op->params.reshape.new_shape[i + 1] != 0) ? ", " : "");
                }
                php_printf(")");
            }
            break;

        case OP_TRANSPOSE:
            if (op->params.transpose.perm[0] != 0)
            {
                php_printf(" [perm=(");
                for (int i = 0; i < MAX_DIMS && op->params.transpose.perm[i] != 0; i++)
                {
                    php_printf("%d%s", op->params.transpose.perm[i],
                               (i < MAX_DIMS - 1 && op->params.transpose.perm[i + 1] != 0) ? ", " : "");
                }
                php_printf(")]");
            }
            break;

        case OP_SLICE:
            php_printf(" [slice]");
            break;
        }

        php_printf("\n");
        current = current->next;
        index++;
    }

    php_printf("\n--- Summary ---\n");
    php_printf("Total operations: %zu\n", context->operation_nodes.count);

    int tensor_tensor_ops = 0, tensor_scalar_ops = 0, scalar_tensor_ops = 0;
    int unary_ops = 0, other_ops = 0;

    current = context->operation_nodes.head;
    while (current != NULL)
    {
        if (current->op)
        {
            switch (current->op->arity)
            {
            case OP_TYPE_TENSOR_TENSOR:
                tensor_tensor_ops++;
                break;
            case OP_TYPE_TENSOR_SCALAR:
                tensor_scalar_ops++;
                break;
            case OP_TYPE_SCALAR_TENSOR:
                scalar_tensor_ops++;
                break;
            case OP_TYPE_UNARY_TENSOR:
                unary_ops++;
                break;
            default:
                other_ops++;
                break;
            }
        }
        current = current->next;
    }

    if (tensor_tensor_ops > 0)
        php_printf("Tensor-Tensor ops: %d\n", tensor_tensor_ops);
    if (tensor_scalar_ops > 0)
        php_printf("Tensor-Scalar ops: %d\n", tensor_scalar_ops);
    if (scalar_tensor_ops > 0)
        php_printf("Scalar-Tensor ops: %d\n", scalar_tensor_ops);
    if (unary_ops > 0)
        php_printf("Unary ops: %d\n", unary_ops);

    php_printf("Tensors created: T0..T%d\n", context->tracker.next_temp_id - 1);

    if (context->tracker.next_input_id > 0)
    {
        php_printf("Inputs: I0..I%d\n", context->tracker.next_input_id - 1);
    }
    if (context->tracker.next_constant_id > 0)
    {
        php_printf("Constants: %d\n", context->tracker.next_constant_id);
    }

    php_printf("Final result: %s\n",
               context->trace_output ? fusion_get_tensor_alias(context->trace_output) : "None");
    php_printf("=======================================\n");
}

void fusion_clear_tensor_trace(tensor_t *tensor)
{
    if (!tensor)
        return;

    tensor->trace.defining_op = NULL;
    tensor->trace.expr_id = -1;
    tensor->trace.expr_alias[0] = '\0';
}

static void fusion_auto_tag_tensor(tensor_t *tensor)
{
    if (!tensor || tensor->trace.expr_alias[0] != '\0')
    {
        return;
    }

    fusion_context_t *context = CUDA_G(current_fusion_context);
    if (!context || !context->tracker.is_active)
    {
        return;
    }

    snprintf(tensor->trace.expr_alias, sizeof(tensor->trace.expr_alias),
             "I%d", context->tracker.next_input_id++);
    tensor->trace.expr_id = -1;
    tensor->trace.defining_op = NULL;
}

static operation_t *fusion_create_base_op(operation_type_t type, tensor_t *result)
{
    fusion_context_t *context = CUDA_G(current_fusion_context);
    if (!context || !context->tracker.is_active)
    {
        return NULL;
    }

    operation_t *op = (operation_t *)emalloc(sizeof(operation_t));
    if (!op)
        return NULL;

    memset(op, 0, sizeof(operation_t));
    op->type = type;

    if (result)
    {
        op->output_ndims = result->ndims;
        int dims_to_copy = result->ndims;
        if (dims_to_copy > MAX_DIMS)
            dims_to_copy = MAX_DIMS;

        for (int i = 0; i < dims_to_copy; i++)
        {
            op->output_shape[i] = result->shape[i];
        }
    }

    op->output_id = context->tracker.next_temp_id++;
    snprintf(op->output_alias, sizeof(op->output_alias), "T%d", op->output_id);

    op_list_add(&context->operation_nodes, op);
    context->tracker.op_counter++;

    if (result)
    {
        fusion_link_tensor(result, op);
    }

    return op;
}