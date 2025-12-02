#include "kernel_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "php.h"

#define MAX_CODE_SIZE 65536
#define MAX_PARAMS 64

static const char *get_cuda_operator(operation_type_t type);
static const char *get_cuda_function(operation_type_t type);
static void add_tensor_if_new(tensor_list_t *list, tensor_t *tensor);
static size_t get_total_elements(const tensor_t *t);
static const char *get_operand_alias(const tensor_t *tensor);
static const char *get_result_alias(const tensor_t *tensor, int idx);

kernel_generator_t *kernel_generator_create(fusion_context_t *context)
{
    if (!context)
        return NULL;

    kernel_generator_t *gen = (kernel_generator_t *)ecalloc(1, sizeof(kernel_generator_t));
    if (!gen)
        return NULL;

    gen->context = context;
    gen->block_size = 256;
    gen->grid_size = 0;

    gen->header_code = (char *)ecalloc(MAX_CODE_SIZE, 1);
    gen->device_code = (char *)ecalloc(MAX_CODE_SIZE, 1);
    gen->kernel_code = (char *)ecalloc(MAX_CODE_SIZE, 1);
    gen->launch_code = (char *)ecalloc(MAX_CODE_SIZE, 1);

    if (!gen->header_code || !gen->device_code || !gen->kernel_code || !gen->launch_code)
    {
        kernel_generator_destroy(gen);
        return NULL;
    }

    return gen;
}

void kernel_generator_destroy(kernel_generator_t *gen)
{
    if (!gen)
        return;

    if (gen->header_code)
        efree(gen->header_code);
    if (gen->device_code)
        efree(gen->device_code);
    if (gen->kernel_code)
        efree(gen->kernel_code);
    if (gen->launch_code)
        efree(gen->launch_code);

    efree(gen);
}

static void add_tensor_if_new(tensor_list_t *list, tensor_t *tensor)
{
    if (!tensor)
        return;

    for (int i = 0; i < list->count; i++)
    {
        if (list->tensors[i] == tensor)
        {
            return;
        }
    }

    if (list->count >= list->capacity)
    {
        list->capacity = list->capacity == 0 ? 4 : list->capacity * 2;
        list->tensors = (tensor_t **)erealloc(list->tensors,
                                              sizeof(tensor_t *) * list->capacity);
    }

    list->tensors[list->count++] = tensor;
}

static size_t get_total_elements(const tensor_t *t)
{
    if (!t || t->ndims == 0)
        return 0;
    size_t total = 1;
    for (int i = 0; i < t->ndims; i++)
    {
        total *= t->shape[i];
    }
    return total;
}

static tensor_list_t collect_unique_tensors(fusion_context_t *ctx)
{
    tensor_list_t list = {0};
    list.capacity = 32;
    list.tensors = (tensor_t **)ecalloc(list.capacity, sizeof(tensor_t *));

    op_list_node_t *current = ctx->operation_nodes.head;
    while (current != NULL)
    {
        operation_t *op = current->op;
        if (op)
        {
            if (op->result)
            {
                add_tensor_if_new(&list, op->result);
            }

            switch (op->arity)
            {
            case OP_TYPE_TENSOR_TENSOR:
                add_tensor_if_new(&list, op->operands.tensor_tensor.a);
                add_tensor_if_new(&list, op->operands.tensor_tensor.b);
                break;
            case OP_TYPE_TENSOR_SCALAR:
                add_tensor_if_new(&list, op->operands.tensor_scalar.tensor);
                break;
            case OP_TYPE_SCALAR_TENSOR:
                add_tensor_if_new(&list, op->operands.scalar_tensor.tensor);
                break;
            case OP_TYPE_UNARY_TENSOR:
                add_tensor_if_new(&list, op->operands.unary.tensor);
                break;
            default:
                break;
            }
        }
        current = current->next;
    }

    if (ctx->trace_output)
    {
        add_tensor_if_new(&list, ctx->trace_output);
    }

    return list;
}

static void classify_tensors(tensor_list_t *all_tensors,
                             tensor_list_t *inputs, tensor_list_t *outputs, tensor_list_t *temps)
{
    inputs->capacity = MAX_PARAMS;
    inputs->tensors = (tensor_t **)ecalloc(inputs->capacity, sizeof(tensor_t *));
    outputs->capacity = MAX_PARAMS;
    outputs->tensors = (tensor_t **)ecalloc(outputs->capacity, sizeof(tensor_t *));
    temps->capacity = MAX_PARAMS;
    temps->tensors = (tensor_t **)ecalloc(temps->capacity, sizeof(tensor_t *));

    for (int i = 0; i < all_tensors->count; i++)
    {
        tensor_t *t = all_tensors->tensors[i];
        if (!t)
            continue;

        switch (t->trace.tensor_type)
        {
        case TENSOR_TYPE_INPUT:
            add_tensor_if_new(inputs, t);
            break;
        case TENSOR_TYPE_OUTPUT:
            add_tensor_if_new(outputs, t);
            break;
        case TENSOR_TYPE_TEMP:
            add_tensor_if_new(temps, t);
            break;
        }
    }
}

static char *generate_parameter_list(
    tensor_list_t *inputs,
    tensor_list_t *outputs)
{
    char *params = (char *)emalloc(MAX_PARAMS * 64);
    params[0] = '\0';
    int param_count = 0;

    for (int i = 0; i < inputs->count; i++)
    {
        tensor_t *t = inputs->tensors[i];
        if (param_count > 0)
        {
            strcat(params, ",\n");
        }

        char param[128];
        snprintf(param, sizeof(param), "    const float* %s", t->trace.expr_alias);
        strcat(params, param);
        param_count++;
    }

    for (int i = 0; i < outputs->count; i++)
    {
        tensor_t *t = outputs->tensors[i];
        if (param_count > 0)
        {
            strcat(params, ",\n");
        }

        char param[128];
        snprintf(param, sizeof(param), "    float* %s", t->trace.expr_alias);
        strcat(params, param);
        param_count++;
    }

    return params;
}

bool kernel_generator_analyze(kernel_generator_t *gen)
{
    if (!gen || !gen->context)
        return false;

    fusion_context_t *ctx = gen->context;

    bool has_reduction = false;
    size_t max_total_elements = 0;

    op_list_node_t *current = ctx->operation_nodes.head;
    while (current != NULL)
    {
        operation_t *op = current->op;
        if (op)
        {
            if (op->type >= OP_REDUCE_SUM && op->type <= OP_REDUCE_PROD)
            {
                has_reduction = true;
            }

            if (op->result && op->result->ndims > 0)
            {
                size_t op_elements = get_total_elements(op->result);
                if (op_elements > max_total_elements)
                {
                    max_total_elements = op_elements;
                }
            }
        }
        current = current->next;
    }

    gen->kernel_type = has_reduction ? KERNEL_TYPE_REDUCTION : KERNEL_TYPE_ELEMENTWISE;
    if (has_reduction && ctx->operation_nodes.count > 1)
    {
        gen->kernel_type = KERNEL_TYPE_MIXED;
    }

    size_t output_elements = 0;
    if (ctx->trace_output)
    {
        output_elements = get_total_elements(ctx->trace_output);
    }

    gen->total_threads = (int)(output_elements > 0 ? output_elements : gen->block_size);

    gen->grid_size = (gen->total_threads + gen->block_size - 1) / gen->block_size;
    if (gen->grid_size == 0)
        gen->grid_size = 1;

    tensor_list_t all_tensors = collect_unique_tensors(ctx);
    tensor_list_t inputs = {0}, outputs = {0}, temps = {0};
    classify_tensors(&all_tensors, &inputs, &outputs, &temps);

    gen->num_params = inputs.count + outputs.count;

    gen->memory_bytes = 0;
    for (int i = 0; i < all_tensors.count; i++)
    {
        if (all_tensors.tensors[i])
        {
            gen->memory_bytes += get_total_elements(all_tensors.tensors[i]) * sizeof(float);
        }
    }

    if (all_tensors.tensors)
        efree(all_tensors.tensors);
    if (inputs.tensors)
        efree(inputs.tensors);
    if (outputs.tensors)
        efree(outputs.tensors);
    if (temps.tensors)
        efree(temps.tensors);

    return true;
}

bool kernel_generator_generate(kernel_generator_t *gen)
{
    if (!gen || !gen->context)
        return false;

    fusion_context_t *ctx = gen->context;

    tensor_list_t all_tensors = collect_unique_tensors(ctx);
    tensor_list_t inputs = {0}, outputs = {0}, temps = {0};
    classify_tensors(&all_tensors, &inputs, &outputs, &temps);

    strcat(gen->header_code, "// Generated CUDA fused kernel\n");
    strcat(gen->header_code, "#include <cmath>\n\n");

    strcat(gen->device_code, "// Device functions\n");
    strcat(gen->device_code, "__device__ float cuda_safe_div(float a, float b) {\n");
    strcat(gen->device_code, "    return b != 0.0f ? a / b : 0.0f;\n");
    strcat(gen->device_code, "}\n\n");

    strcat(gen->kernel_code, "__global__ void fused_kernel(\n");
    char *param_list = generate_parameter_list(&inputs, &outputs);
    strcat(gen->kernel_code, param_list);
    strcat(gen->kernel_code, "\n) {\n");
    efree(param_list);

    strcat(gen->kernel_code, "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n");

    if (gen->total_threads > 0 && (size_t)gen->total_threads < (size_t)gen->block_size * gen->grid_size)
    {
        char bounds_check[256];
        snprintf(bounds_check, sizeof(bounds_check),
                 "    if (idx >= %d) return;\n\n", gen->total_threads);
        strcat(gen->kernel_code, bounds_check);
    }

    strcat(gen->kernel_code, "    // Load inputs from global memory (I0[idx] -> val_I0)\n");
    for (int i = 0; i < inputs.count; i++)
    {
        tensor_t *t = inputs.tensors[i];
        const char *alias = t->trace.expr_alias;

        char variable[128];
        snprintf(variable, sizeof(variable), "    float val_%s = %s[idx];\n", alias, alias);
        strcat(gen->kernel_code, variable);
    }
    strcat(gen->kernel_code, "\n");

    op_list_node_t *current = ctx->operation_nodes.head;
    while (current != NULL)
    {
        operation_t *op = current->op;
        if (!op)
        {
            current = current->next;
            continue;
        }

        char line[512];
        
        const char *result_expr = get_result_alias(op->result, gen->total_threads);

        const char *var_type = op->result->trace.tensor_type == TENSOR_TYPE_TEMP ? "float " : "";
        const char *assign_op = op->result->trace.tensor_type == TENSOR_TYPE_TEMP ? " = " : " = ";

        switch (op->arity)
        {
        case OP_TYPE_TENSOR_TENSOR:
        {
            const char *a_alias = get_operand_alias(op->operands.tensor_tensor.a);
            const char *b_alias = get_operand_alias(op->operands.tensor_tensor.b);

            if (op->type == OP_POW)
            {
                snprintf(line, sizeof(line), "    %s %s%s powf(%s, %s);\n",
                         var_type, result_expr, assign_op, a_alias, b_alias);
            }
            else
            {
                snprintf(line, sizeof(line), "    %s %s%s %s %s %s;\n",
                         var_type, result_expr, assign_op, a_alias,
                         get_cuda_operator(op->type),
                         b_alias);
            }
            break;
        }

        case OP_TYPE_TENSOR_SCALAR:
        {
            const char *tensor_alias = get_operand_alias(op->operands.tensor_scalar.tensor);

            if (op->type == OP_POW)
            {
                snprintf(line, sizeof(line), "    %s %s%s powf(%s, %.6ff);\n",
                         var_type, result_expr, assign_op, tensor_alias, op->operands.tensor_scalar.scalar);
            }
            else
            {
                snprintf(line, sizeof(line), "    %s %s%s %s %s %.6ff;\n",
                         var_type, result_expr, assign_op, tensor_alias,
                         get_cuda_operator(op->type),
                         op->operands.tensor_scalar.scalar);
            }
            break;
        }

        case OP_TYPE_SCALAR_TENSOR:
        {
            const char *tensor_alias = get_operand_alias(op->operands.scalar_tensor.tensor);

            if (op->type == OP_SUB)
            {
                snprintf(line, sizeof(line), "    %s %s%s %.6ff - %s;\n",
                         var_type, result_expr, assign_op, op->operands.scalar_tensor.scalar, tensor_alias);
            }
            else if (op->type == OP_DIV)
            {
                snprintf(line, sizeof(line), "    %s %s%s cuda_safe_div(%.6ff, %s);\n",
                         var_type, result_expr, assign_op, op->operands.scalar_tensor.scalar, tensor_alias);
            }
            else if (op->type == OP_POW)
            {
                snprintf(line, sizeof(line), "    %s %s%s powf(%.6ff, %s);\n",
                         var_type, result_expr, assign_op, op->operands.scalar_tensor.scalar, tensor_alias);
            }
            else
            {
                snprintf(line, sizeof(line), "    %s %s%s %.6ff %s %s;\n",
                         var_type, result_expr, assign_op, op->operands.scalar_tensor.scalar,
                         get_cuda_operator(op->type),
                         tensor_alias);
            }
            break;
        }

        case OP_TYPE_UNARY_TENSOR:
        {
            const char *tensor_alias = get_operand_alias(op->operands.unary.tensor);
            const char *cuda_func = get_cuda_function(op->type);

            if (op->type == OP_NEG)
            {
                snprintf(line, sizeof(line), "    %s %s%s -%s;\n",
                         var_type, result_expr, assign_op, tensor_alias);
            }
            else
            {
                snprintf(line, sizeof(line), "    %s %s%s %s(%s);\n",
                         var_type, result_expr, assign_op, cuda_func, tensor_alias);
            }
            break;
        }

        default:
            snprintf(line, sizeof(line), "    // Unsupported operation type\n");
            break;
        }

        strcat(gen->kernel_code, line);
        current = current->next;
    }

    strcat(gen->kernel_code, "}\n");

    snprintf(gen->launch_code, MAX_CODE_SIZE,
             "// Launch configuration\n"
             "dim3 blockDim(%d);\n"
             "dim3 gridDim(%d);\n\n"
             "// Launch kernel\n"
             "fused_kernel<<<gridDim, blockDim>>>(\n",
             gen->block_size, gen->grid_size);

    int launch_count = 0;
    for (int i = 0; i < inputs.count; i++)
    {
        char param[128];
        snprintf(param, sizeof(param), "    %s%s", inputs.tensors[i]->trace.expr_alias,
                 (launch_count < gen->num_params - 1) ? ",\n" : "");
        strcat(gen->launch_code, param);
        launch_count++;
    }

    for (int i = 0; i < outputs.count; i++)
    {
        char param[128];
        snprintf(param, sizeof(param), "    %s%s", outputs.tensors[i]->trace.expr_alias,
                 (launch_count < gen->num_params - 1) ? ",\n" : "\n");
        strcat(gen->launch_code, param);
        launch_count++;
    }

    strcat(gen->launch_code, ");\n");
    strcat(gen->launch_code, "cudaDeviceSynchronize();\n");

    if (all_tensors.tensors)
        efree(all_tensors.tensors);
    if (inputs.tensors)
        efree(inputs.tensors);
    if (outputs.tensors)
        efree(outputs.tensors);
    if (temps.tensors)
        efree(temps.tensors);

    return true;
}

static const char *get_operand_alias(const tensor_t *tensor)
{
    if (!tensor || tensor->trace.expr_alias[0] == '\0')
    {
        return "?";
    }

    if (tensor->trace.tensor_type == TENSOR_TYPE_TEMP)
    {
        return tensor->trace.expr_alias;
    }

    if (tensor->trace.tensor_type == TENSOR_TYPE_INPUT)
    {
        char *buffer = (char *)emalloc(16);
        snprintf(buffer, 16, "val_%s", tensor->trace.expr_alias);
        return buffer;
    }

    return "?";
}

static const char *get_result_alias(const tensor_t *tensor, int total_threads)
{
    if (!tensor || tensor->trace.expr_alias[0] == '\0')
    {
        return "?";
    }

    if (tensor->trace.tensor_type == TENSOR_TYPE_TEMP)
    {
        return tensor->trace.expr_alias;
    }

    if (tensor->trace.tensor_type == TENSOR_TYPE_OUTPUT)
    {
        char *buffer = (char *)emalloc(16); 
        snprintf(buffer, 16, "%s[idx]", tensor->trace.expr_alias);
        return buffer;
    }

    return "?";
}

static const char *get_cuda_operator(operation_type_t type)
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
    case OP_GT:
        return ">";
    case OP_LT:
        return "<";
    case OP_EQ:
        return "==";
    default:
        return "?";
    }
}

static const char *get_cuda_function(operation_type_t type)
{
    switch (type)
    {
    case OP_EXP:
        return "expf";
    case OP_SQRT:
        return "sqrtf";
    case OP_LOG:
        return "logf";
    case OP_SIN:
        return "sinf";
    case OP_COS:
        return "cosf";
    case OP_TAN:
        return "tanf";
    case OP_ABS:
        return "fabsf";
    default:
        return "?";
    }
}

void kernel_generator_print(kernel_generator_t *gen)
{
    if (!gen)
        return;

    php_printf("\n=== GENERATED CUDA KERNEL ===\n\n");

    php_printf("// Kernel type: ");
    switch (gen->kernel_type)
    {
    case KERNEL_TYPE_ELEMENTWISE:
        php_printf("Elementwise\n");
        break;
    case KERNEL_TYPE_REDUCTION:
        php_printf("Reduction\n");
        break;
    case KERNEL_TYPE_MIXED:
        php_printf("Mixed\n");
        break;
    }

    php_printf("// Threads: %d (%d blocks x %d threads)\n",
               gen->total_threads, gen->grid_size, gen->block_size);
    php_printf("// Memory: ~%d bytes\n", gen->memory_bytes);
    php_printf("// Parameters: %d\n\n", gen->num_params);

    php_printf("// HEADER\n%s\n", gen->header_code);
    php_printf("// DEVICE FUNCTIONS\n%s\n", gen->device_code);
    php_printf("// KERNEL\n%s\n", gen->kernel_code);
    php_printf("// LAUNCH CONFIG\n%s\n", gen->launch_code);
    php_printf("=======================================\n");
}


bool kernel_generator_compile(kernel_generator_t *gen)
{
    // @todo
    return true;
}

bool kernel_generator_execute(kernel_generator_t *gen)
{
    // @todo
    return true;
}