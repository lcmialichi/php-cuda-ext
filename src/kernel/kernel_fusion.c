#include "php.h"

#ifdef ZTS
#include "TSRM.h"
#endif

#include "cuda_globals.h"
#include "tensor.h"
#include "operations.h"
#include <string.h>
#include "php.h"

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

    CUDA_G(current_fusion_context) = context;
    CUDA_G(is_tracing_enabled) = true;
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
                if (current->op->input_a)
                {
                }
                if (current->op->input_b)
                {
                }
                efree(current->op);
            }

            efree(current);
            current = next;
        }

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
    }
    else
    {
        php_error_docref(NULL, E_WARNING, "set_current_trace_output called when tracing is inactive (TLS context is NULL).");
    }
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

static const char *operation_type_to_string(operation_type_t type)
{
    switch (type)
    {
    case OP_ADD:
        return "ADD";
    case OP_SUB:
        return "SUB";
    case OP_MUL:
        return "MUL";
    case OP_DIV:
        return "DIV";
    case OP_POW:
        return "POW";
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

void op_list_print(op_list_t *list)
{
    php_printf("\n--- Kernel Fusion Trace (Total: %zu ops) ---\n", list->count);

    op_list_node_t *current = list->head;
    int index = 0;

    while (current != NULL)
    {
        operation_t *op = current->op;
        const char *op_name = operation_type_to_string(op->type);

        php_printf("[%d] Operation: %s\n", index, op_name);
        char output_shape_str[256] = "";
        if (op->output_ndims > 0)
        {

            php_printf("    -> Output Shape: (Dim: %d) [", op->output_ndims);
            for (int i = 0; i < op->output_ndims; i++)
            {
                php_printf("%d%s", op->output_shape[i], (i == op->output_ndims - 1) ? "" : ", ");
            }
            php_printf("]\n");
        }

        if (op->input_a)
        {
            if (op->input_a->ndims > 0)
            {
                php_printf("    -> Input A Shape: %s", tensor_shape_as_string(op->input_a));
            }
            else
            {
                php_printf("    -> Input A Shape: [SCALAR]");
            }

            php_printf(" (Type: %d)\n", op->input_a->dtype);
        }

        if (op->input_b)
        {
            if (op->input_b->ndims > 0)
            {
                php_printf("    -> Input B Shape: %s", tensor_shape_as_string(op->input_b));
            }
            else
            {
                php_printf("    -> Input B Shape: [SCALAR]");
            }

            php_printf(" (Type: %d)\n", op->input_b->dtype);
        }

        if (op->type >= OP_REDUCE_SUM && op->type <= OP_REDUCE_PROD)
        {
            php_printf("    -> Reduction Params: Axis=%d, KeepDims=%s\n",
                       op->data.reduction.axis, op->data.reduction.keep_dims ? "True" : "False");
        }

        php_printf("---------------------------------------------------\n");

        current = current->next;
        index++;
    }

    php_printf("--- End of Trace ---\n");
}