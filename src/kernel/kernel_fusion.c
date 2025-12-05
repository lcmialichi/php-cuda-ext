#include "php.h"
#include "kernel_generator.h"
#include "cuda_globals.h"
#include "tensor.h"
#include "operations.h"
#include <string.h>

#ifdef ZTS
#include "TSRM.h"
#endif

static void fusion_auto_tag_tensor(tensor_t *tensor);
static operation_t *fusion_create_base_op(operation_type_t type, tensor_t *result);
static kernel_model_t get_operation_model(operation_type_t op_type);

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
    context->tracker.temp_id_count = 0;
    context->tracker.input_id_count = 0;
    context->tracker.constant_id_count = 0;
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
                 "%s%d", const_type, context->tracker.constant_id_count++);
    }
    else
    {
        snprintf(tensor->trace.expr_alias, sizeof(tensor->trace.expr_alias),
                 "C%d", context->tracker.constant_id_count++);
    }

    tensor->trace.expr_id = -2;
    tensor->trace.defining_op = NULL;
    tensor->trace.defining_op = NULL;
    tensor->trace.tensor_type = TENSOR_TYPE_INPUT;
}

operation_t *fusion_create_tensor_tensor_op(operation_type_t type,
                                            tensor_t *a, tensor_t *b,
                                            tensor_t *result)
{
    TENSOR_KERNEL_TRACE(a);
    TENSOR_KERNEL_TRACE(b);
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

    TENSOR_KERNEL_TRACE(tensor);
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

    TENSOR_KERNEL_TRACE(tensor);
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

    TENSOR_KERNEL_TRACE(tensor);
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

void stop_kernel_fusions()
{
    CUDA_G(is_tracing_enabled) = false;

    fusion_context_t *context = CUDA_G(current_fusion_context);

    if (context != NULL)
    {
        op_list_node_t *current = context->operation_nodes.head;
        while (current != NULL)
        {
            op_list_node_t *next = current->next;
            operation_t *op = current->op;

            if (op == NULL)
            {
                efree(current);
                current = next;
            }

            switch (op->arity)
            {
            case OP_TYPE_TENSOR_TENSOR:
                if (op->operands.tensor_tensor.a)
                    TENSOR_DEL_REF(op->operands.tensor_tensor.a);
                if (op->operands.tensor_tensor.b)
                    TENSOR_DEL_REF(op->operands.tensor_tensor.b);
                break;
            case OP_TYPE_TENSOR_SCALAR:
                if (op->operands.tensor_scalar.tensor)
                    TENSOR_DEL_REF(op->operands.tensor_scalar.tensor);
                break;
            case OP_TYPE_SCALAR_TENSOR:
                if (op->operands.scalar_tensor.tensor)
                    TENSOR_DEL_REF(op->operands.scalar_tensor.tensor);
                break;
            case OP_TYPE_UNARY_TENSOR:
                if (op->operands.unary.tensor)
                    TENSOR_DEL_REF(op->operands.unary.tensor);
                break;
            default:
                break;
            }

            efree(op);

            efree(current);
            current = next;
        }

        efree(context);
        CUDA_G(current_fusion_context) = NULL;
    }
}

bool is_tracing()
{
    return CUDA_G(is_tracing_enabled);
}

tensor_t *compile_and_execute_fusion(tensor_t *tensor)
{
    if (!tensor)
        return NULL;

    printf("1\n");
    multi_kernel_generator *gen = multi_kernel_create(tensor);
    if (!gen)
        return NULL;

    printf("2\n");
    if (!multi_kernel_analyze_and_split(gen))
    {
        multi_kernel_destroy(gen);
        return NULL;
    }

    printf("3\n");
    if (!multi_kernel_generate(gen))
    {
        multi_kernel_destroy(gen);
        return NULL;
    }
    printf("4\n");
    multi_kernel_generator_print(gen);
    multi_kernel_destroy(gen);
    printf("5\n");
    return tensor;
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
             "I%d", context->tracker.input_id_count++);
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
        result->trace.tensor_type = TENSOR_TYPE_TEMP;
        op->result = result;
        op->output_ndims = result->ndims;
        int dims_to_copy = result->ndims;
        if (dims_to_copy > MAX_DIMS)
            dims_to_copy = MAX_DIMS;

        for (int i = 0; i < dims_to_copy; i++)
        {
            op->output_shape[i] = result->shape[i];
        }
    }

    op->output_id = context->tracker.temp_id_count++;
    op->model = get_operation_model(op->type);
    snprintf(op->output_alias, sizeof(op->output_alias), "T%d", op->output_id);

    op_list_add(&context->operation_nodes, op);
    context->tracker.op_counter++;

    if (result)
    {
        fusion_link_tensor(result, op);
    }

    return op;
}

static kernel_model_t get_operation_model(operation_type_t op_type)
{
    switch (op_type)
    {
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV:
    case OP_POW:
    case OP_EXP:
    case OP_SQRT:
    case OP_LOG:
    case OP_SIN:
    case OP_COS:
    case OP_TAN:
    case OP_ABS:
    case OP_NEG:
    case OP_SELECT:
    case OP_CLAMP:
    case OP_CEIL:
    case OP_FLOOR:
    case OP_ROUND:
    case OP_GT:
    case OP_LT:
    case OP_EQ:
    case OP_NE:
    case OP_GE:
    case OP_LE:
        return MODEL_ELEMENT_WISE;

    case OP_REDUCE_SUM:
    case OP_REDUCE_MEAN:
    case OP_REDUCE_MAX:
    case OP_REDUCE_MIN:
    case OP_REDUCE_PROD:
    case OP_ARG_MAX:
    case OP_ARG_MIN:
        return MODEL_REDUCTION;

    case OP_RESHAPE:
    case OP_TRANSPOSE:
    case OP_SLICE:
        return METADATA_TRANSFORM;

    case OP_CONCAT:
        return MODEL_CONCAT;
    case OP_MATMUL:
        return MODEL_COMPUTE_CALL;

    default:
        return MODEL_COMPUTE_CALL;
    }
}