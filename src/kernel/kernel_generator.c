#include "kernel_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include "php.h"

#define MAX_CODE_SIZE 65536
#define MAX_PARAMS 64

static const char *get_result_alias(const tensor_t *tensor);
static const char *get_operand_alias(const tensor_t *tensor);
static const char *get_cuda_operator(operation_type_t type);
static const char *get_cuda_function(operation_type_t type);
static size_t get_total_elements(const tensor_t *t);
static void add_tensor_if_new(tensor_list_t *list, tensor_t *tensor);

static bool op_is_in_list(op_list_t *list, operation_t *op);
static void op_list_destroy(op_list_t *list);

static void kernel_gen_headers(kernel_generator_t *gen);
static void kernel_gen_device_funcs(kernel_generator_t *gen);
static void kernel_gen_kernels();
static void kernel_gen_launchers();

static char *kernel_define_op(operation_t *op);

static void kernel_define_binary_op(operation_t *op, const char *result_expr,
                                    const char *func, const char *oper,
                                    char *expr, size_t expr_size);
static void kernel_define_t_scalar_op(operation_t *op, const char *result_expr,
                                      const char *func, const char *oper,
                                      char *expr, size_t expr_size);
static void kernel_define_scalar_t_op(operation_t *op, const char *result_expr,
                                      const char *func, const char *oper,
                                      char *expr, size_t expr_size);
static void kernel_define_unary_op(operation_t *op, const char *result_expr,
                                   const char *func, const char *oper,
                                   char *expr, size_t expr_size);
typedef void (*op_definer_t)(operation_t *op, const char *result_expr,
                             const char *func, const char *oper,
                             char *expr, size_t expr_size);

static const op_definer_t op_definers[] = {
    [OP_TYPE_TENSOR_TENSOR] = kernel_define_binary_op,
    [OP_TYPE_TENSOR_SCALAR] = kernel_define_t_scalar_op,
    [OP_TYPE_SCALAR_TENSOR] = kernel_define_scalar_t_op,
    [OP_TYPE_UNARY_TENSOR] = kernel_define_unary_op,
};

static void append_code(char **code_buffer, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(*code_buffer + strlen(*code_buffer), MAX_CODE_SIZE - strlen(*code_buffer), format, args);
    va_end(args);
}

static void op_list_init(op_list_t *list)
{
    list->head = NULL;
    list->tail = NULL;
    list->count = 0;
}

static void op_list_append(op_list_t *list, operation_t *op)
{
    op_list_node_t *new_node = (op_list_node_t *)ecalloc(1, sizeof(op_list_node_t));
    new_node->op = op;
    new_node->next = NULL;

    if (list->tail == NULL)
    {
        list->head = new_node;
        list->tail = new_node;
    }
    else
    {
        list->tail->next = new_node;
        list->tail = new_node;
    }
    list->count++;
}

static bool op_is_in_list(op_list_t *list, operation_t *op)
{
    op_list_node_t *current = list->head;
    while (current != NULL)
    {
        if (current->op == op)
            return true;
        current = current->next;
    }
    return false;
}

static void op_list_destroy(op_list_t *list)
{
    op_list_node_t *current = list->head;
    op_list_node_t *next;
    while (current != NULL)
    {
        next = current->next;
        efree(current);
        current = next;
    }
    list->head = NULL;
    list->tail = NULL;
    list->count = 0;
}

static void tensor_list_init(tensor_list_t *list)
{
    memset(list, 0, sizeof(tensor_list_t));
}

static void tensor_list_destroy(tensor_list_t *list)
{
    if (list->tensors)
        efree(list->tensors);
    memset(list, 0, sizeof(tensor_list_t));
}

static int get_op_inputs(operation_t *op, tensor_t *inputs[])
{
    switch (op->arity)
    {
    case OP_TYPE_TENSOR_TENSOR:
        inputs[0] = op->operands.tensor_tensor.a;
        inputs[1] = op->operands.tensor_tensor.b;
        return 2;
    case OP_TYPE_TENSOR_SCALAR:
        inputs[0] = op->operands.tensor_scalar.tensor;
        return 1;
    case OP_TYPE_SCALAR_TENSOR:
        inputs[0] = op->operands.scalar_tensor.tensor;
        return 1;
    case OP_TYPE_UNARY_TENSOR:
        inputs[0] = op->operands.unary.tensor;
        return 1;
    default:
        return 0;
    }
}

kernel_generator_t *kernel_generator_create(tensor_t *final_output)
{
    if (!final_output)
        return NULL;

    kernel_generator_t *gen = (kernel_generator_t *)ecalloc(1, sizeof(kernel_generator_t));
    if (!gen)
        return NULL;

    gen->final_output = final_output;
    gen->block_size = 256;
    gen->grid_size = 0;

    op_list_init(&gen->required_ops);
    tensor_list_init(&gen->inputs);
    tensor_list_init(&gen->outputs);
    tensor_list_init(&gen->temps);

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

    op_list_destroy(&gen->required_ops);
    tensor_list_destroy(&gen->inputs);
    tensor_list_destroy(&gen->outputs);
    tensor_list_destroy(&gen->temps);

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

static void collect_dependencies_recursive(operation_t *op, op_list_t *required_ops)
{
    if (op == NULL || op_is_in_list(required_ops, op))
    {
        return;
    }

    tensor_t *inputs[2] = {NULL};

    int num_inputs = get_op_inputs(op, inputs);

    for (int i = 0; i < num_inputs; i++)
    {
        if (inputs[i] != NULL && inputs[i]->trace.defining_op != NULL)
        {
            collect_dependencies_recursive(inputs[i]->trace.defining_op, required_ops);
        }
    }

    op_list_append(required_ops, op);
}

static void collect_unique_tensors(const op_list_t *required_ops, tensor_list_t *list)
{
    list->capacity = 32;
    list->tensors = (tensor_t **)ecalloc(list->capacity, sizeof(tensor_t *));

    operation_t *op;
    KERNEL_OPLIST_FOREACH(required_ops->head, op)
    {
        if (!op)
        {
            continue;
        }

        if (op->result)
        {
            add_tensor_if_new(list, op->result);
        }
        tensor_t *inputs[2] = {NULL};
        int num_inputs = get_op_inputs(op, inputs);
        for (int i = 0; i < num_inputs; i++)
        {
            add_tensor_if_new(list, inputs[i]);
        }
    }
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

        if (t->trace.defining_op == NULL)
        {
            add_tensor_if_new(inputs, t);
            continue;
        }

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

bool kernel_generator_analyze(kernel_generator_t *gen)
{
    if (!gen || !gen->final_output)
        return false;

    op_list_destroy(&gen->required_ops);
    op_list_init(&gen->required_ops);

    snprintf(gen->final_output->trace.expr_alias, sizeof("__out"),
             "%s", "__out");

    gen->final_output->trace.tensor_type = TENSOR_TYPE_OUTPUT;

    if (gen->final_output->trace.defining_op)
    {
        collect_dependencies_recursive(gen->final_output->trace.defining_op, &gen->required_ops);
    }
    else
    {
        gen->kernel_type = KERNEL_TYPE_ELEMENTWISE;
        gen->total_threads = (int)get_total_elements(gen->final_output);
        gen->grid_size = (gen->total_threads + gen->block_size - 1) / gen->block_size;
        return true;
    }

    bool has_reduction = false;
    op_list_node_t *current = gen->required_ops.head;
    size_t max_total_elements = 0;

    operation_t *op;
    KERNEL_OPLIST_FOREACH(gen->required_ops.head, op)
    {
        if (!op)
        {
            continue;
        }

        if (op->type >= OP_REDUCE_SUM && op->type <= OP_REDUCE_PROD)
        {
            has_reduction = true;
        }
        if (op->result && op->result->ndims > 0)
        {
            size_t op_elements = get_total_elements(op->result);
            if (op_elements > max_total_elements)
                max_total_elements = op_elements;
        }
    }

    gen->kernel_type = has_reduction ? KERNEL_TYPE_REDUCTION : KERNEL_TYPE_ELEMENTWISE;
    if (has_reduction && gen->required_ops.count > 1)
        gen->kernel_type = KERNEL_TYPE_MIXED;

    size_t output_elements = get_total_elements(gen->final_output);
    gen->total_threads = (int)(output_elements > 0 ? output_elements : gen->block_size);
    gen->grid_size = (gen->total_threads + gen->block_size - 1) / gen->block_size;
    if (gen->grid_size == 0)
        gen->grid_size = 1;

    tensor_list_t all_tensors = {0};
    collect_unique_tensors(&gen->required_ops, &all_tensors);

    tensor_list_destroy(&gen->inputs);
    tensor_list_destroy(&gen->outputs);
    tensor_list_destroy(&gen->temps);

    classify_tensors(&all_tensors, &gen->inputs, &gen->outputs, &gen->temps);

    // add_tensor_if_new(&gen->outputs, gen->final_output);

    gen->num_params = gen->inputs.count + gen->outputs.count;

    gen->memory_bytes = 0;
    for (int i = 0; i < all_tensors.count; i++)
    {
        if (all_tensors.tensors[i])
        {
            gen->memory_bytes += get_total_elements(all_tensors.tensors[i]) * sizeof(float);
        }
    }

    tensor_list_destroy(&all_tensors);
    return true;
}

static char *generate_parameter_list(tensor_list_t *inputs, tensor_list_t *outputs)
{
    char *params = (char *)ecalloc(MAX_CODE_SIZE, 1);
    int param_count = 0;

    for (int i = 0; i < inputs->count; i++)
    {
        tensor_t *t = inputs->tensors[i];
        if (param_count > 0)
        {
            append_code(&params, ",\n");
        }
        append_code(&params, "  const float* %s", t->trace.expr_alias);
        param_count++;
    }

    for (int i = 0; i < outputs->count; i++)
    {
        tensor_t *t = outputs->tensors[i];
        if (param_count > 0)
        {
            append_code(&params, ",\n");
        }
        append_code(&params, "  float* %s", t->trace.expr_alias);
        param_count++;
    }

    return params;
}

bool kernel_generator_generate(kernel_generator_t *gen)
{
    if (!gen || !gen->final_output)
        return false;

    tensor_list_t *inputs = &gen->inputs;
    tensor_list_t *outputs = &gen->outputs;
    tensor_list_t *temps = &gen->temps;

    kernel_gen_headers(gen);
    kernel_gen_device_funcs(gen);

    append_code(&gen->kernel_code, "__global__ void fused_kernel(\n");
    char *param_list = generate_parameter_list(inputs, outputs);
    append_code(&gen->kernel_code, "%s", param_list);
    append_code(&gen->kernel_code, "\n) {\n");
    efree(param_list);

    append_code(&gen->kernel_code, "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n");

    if (gen->total_threads > 0 && (size_t)gen->total_threads < (size_t)gen->block_size * gen->grid_size)
    {
        append_code(&gen->kernel_code, "  if (idx >= %d) return;\n\n", gen->total_threads);
    }

    append_code(&gen->kernel_code, "  // Virtual registers for temporary results\n");
    for (int i = 0; i < temps->count; i++)
    {
        append_code(&gen->kernel_code, "  float temp_%s;\n", temps->tensors[i]->trace.expr_alias);
    }
    append_code(&gen->kernel_code, "\n");
    append_code(&gen->kernel_code, "  // Fused operations (Topologically Sorted)\n");

    operation_t *op;
    KERNEL_OPLIST_FOREACH(gen->required_ops.head, op)
    {
        char *expr = kernel_define_op(op);
        if (expr)
        {
            append_code(&gen->kernel_code, "%s\n", expr);
            efree(expr);
        }
        // @todo if could not generate a operation expr return error
    }

    append_code(&gen->kernel_code, "}\n");

    append_code(&gen->launch_code,
                "// Launch configuration\n"
                "dim3 blockDim(%d);\n"
                "dim3 gridDim(%d);\n\n"
                "// Launch kernel\n"
                "fused_kernel<<<gridDim, blockDim>>>(\n",
                gen->block_size, gen->grid_size);

    int launch_count = 0;
    for (int i = 0; i < inputs->count; i++)
    {
        append_code(&gen->launch_code, "  %s%s", inputs->tensors[i]->trace.expr_alias,
                    (launch_count < gen->num_params - 1) ? ",\n" : "");
        launch_count++;
    }

    for (int i = 0; i < outputs->count; i++)
    {
        append_code(&gen->launch_code, "  %s%s", outputs->tensors[i]->trace.expr_alias,
                    (launch_count < gen->num_params - 1) ? ",\n" : "\n");
        launch_count++;
    }

    append_code(&gen->launch_code, ");\ncudaDeviceSynchronize();\n");

    tensor_list_destroy(&gen->inputs);
    tensor_list_destroy(&gen->outputs);
    tensor_list_destroy(&gen->temps);

    return true;
}

static const char *get_operand_alias(const tensor_t *tensor)
{
    if (!tensor || tensor->trace.expr_alias[0] == '\0')
        return "?";

    return tensor->trace.expr_alias;
}

static const char *get_result_alias(const tensor_t *tensor)
{
    if (!tensor || tensor->trace.expr_alias[0] == '\0')
        return "?";

    return tensor->trace.expr_alias;
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

static void kernel_gen_headers(kernel_generator_t *gen)
{
    append_code(&gen->header_code,
                "// Generated CUDA fused kernel\n#include <cmath>\n\n");
}

static void kernel_gen_device_funcs(kernel_generator_t *gen)
{
    append_code(&gen->device_code,
                "// Device functions\n",
                "__device__ float cuda_safe_div(float a, float b) {\n",
                "  return b != 0.0f ? a / b : 0.0f;\n}\n\n");
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

static char *kernel_define_op(operation_t *op)
{
    if (!op || op->arity >= sizeof(op_definers) / sizeof(op_definers[0]))
    {
        return estrdup("  // Invalid operation");
    }

    const char *func = get_cuda_function(op->type);
    const char *operator = get_cuda_operator(op->type);

    char r_expr[64] = {0};
    char expr[512] = {0};

    SET_RESULT_ACCESS(r_expr, op->result);

    op_definers[op->arity](op, r_expr, func, operator, expr, sizeof(expr));

    return estrdup(expr);
}

static void kernel_define_binary_op(operation_t *op, const char *result_expr,
                                    const char *func, const char *oper,
                                    char *expr, size_t expr_size)
{
    char a_expr[64] = {0}, b_expr[64] = {0};

    SET_TENSOR_ACCESS(a_expr, op->operands.tensor_tensor.a);
    SET_TENSOR_ACCESS(b_expr, op->operands.tensor_tensor.b);

    if (func)
    {
        snprintf(expr, expr_size, "  %s = %s(%s, %s);",
                 result_expr, func, a_expr, b_expr);
        return;
    }

    if (oper)
    {
        snprintf(expr, expr_size, "  %s = %s %s %s;",
                 result_expr, a_expr, oper, b_expr);
        return;
    }

    snprintf(expr, expr_size, "  // Unsupported binary operation\n");
}

static void kernel_define_scalar_t_op(operation_t *op, const char *result_expr,
                                      const char *func, const char *oper,
                                      char *expr, size_t expr_size)
{
    char t_expr[64] = {0};
    float scalar;

    SET_TENSOR_ACCESS(t_expr, op->operands.scalar_tensor.tensor);
    scalar = op->operands.scalar_tensor.scalar;

    if (func)
    {
        snprintf(expr, expr_size, "  %s = %s(%.6ff, %s);",
                 result_expr, func, scalar, t_expr);
        return;
    }

    if (oper)
    {
        snprintf(expr, expr_size, "  %s = %.6ff %s %s;",
                 result_expr, scalar, oper, t_expr);

        return;
    }

    snprintf(expr, expr_size, "  // Unsupported scalar operation\n");
}

static void kernel_define_t_scalar_op(operation_t *op, const char *result_expr,
                                      const char *func, const char *oper,
                                      char *expr, size_t expr_size)
{
    char t_expr[64] = {0};
    float scalar;

    SET_TENSOR_ACCESS(t_expr, op->operands.tensor_scalar.tensor);
    scalar = op->operands.tensor_scalar.scalar;

    if (func)
    {
        snprintf(expr, expr_size, "  %s = %s(%s, %.6ff);",
                 result_expr, func, t_expr, scalar);
        return;
    }

    if (oper)
    {
        snprintf(expr, expr_size, "  %s = %s %s %.6ff;",
                 result_expr, t_expr, oper, scalar);

        return;
    }

    snprintf(expr, expr_size, "  // Unsupported scalar operation\n");
}

static void kernel_define_unary_op(operation_t *op, const char *result_expr,
                                   const char *func, const char *oper,
                                   char *expr, size_t expr_size)
{
    char t_expr[64] = {0};
    SET_TENSOR_ACCESS(t_expr, op->operands.unary.tensor);

    if (op->type == OP_NEG)
    {
        snprintf(expr, expr_size, "  %s = -%s;",
                 result_expr, t_expr);

        return;
    }

    if (func)
    {
        snprintf(expr, expr_size, "  %s = %s(%s);",
                 result_expr, func, t_expr);
        return;
    }

    snprintf(expr, expr_size, "  // Unsupported unary operation\n");
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
    case OP_POW:
        return "powf";
    default:
        return NULL;
    }
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
        return NULL;
    }
}
