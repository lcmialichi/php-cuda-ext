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

static bool analyze_single_kernel(kernel_generator_t *kg);
static void identify_inter_kernel_connections(multi_kernel_generator *multi);
static void calculate_kernel_dependencies(multi_kernel_generator *multi);
static void collect_all_tensors_from_kernel(kernel_generator_t *kg, tensor_list_t *list);

static void kernel_generator_destroy(kernel_generator_t *gen);

static const char *get_result_alias(const tensor_t *tensor);
static const char *get_operand_alias(const tensor_t *tensor);
static const char *get_cuda_operator(operation_type_t type);
static const char *get_cuda_function(operation_type_t type);
static size_t get_total_elements(const tensor_t *t);
static void add_tensor_if_new(tensor_list_t *list, tensor_t *tensor);

static bool op_is_in_list(op_list_t *list, operation_t *op);
static void op_list_destroy(op_list_t *list);

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

multi_kernel_generator *multi_kernel_create(tensor_t *final_output)
{
    if (!final_output)
        return NULL;

    multi_kernel_generator *multi = (multi_kernel_generator *)ecalloc(1, sizeof(multi_kernel_generator));
    if (!multi)
        return NULL;

    multi->final_output = final_output;
    multi->default_block_size = 256;
    multi->kernel_capacity = 8;
    multi->kernel_count = 0;
    multi->analyzed = 0;
    multi->generated = 0;

    multi->kernels = (kernel_generator_t **)ecalloc(multi->kernel_capacity, sizeof(kernel_generator_t *));

    tensor_list_init(&multi->inter_kernel_tensors);

    multi->header_code = (char *)ecalloc(MAX_CODE_SIZE, 1);
    multi->device_code = (char *)ecalloc(MAX_CODE_SIZE, 1);
    multi->combined_kernel_code = (char *)ecalloc(MAX_CODE_SIZE, 1);
    multi->combined_launch_code = (char *)ecalloc(MAX_CODE_SIZE, 1);

    if (!multi->kernels || !multi->header_code || !multi->device_code ||
        !multi->combined_kernel_code || !multi->combined_launch_code)
    {
        multi_kernel_destroy(multi);
        return NULL;
    }

    return multi;
}

kernel_generator_t *kernel_generator_create_single(tensor_t *final_output, int block_size)
{
    kernel_generator_t *gen = (kernel_generator_t *)ecalloc(1, sizeof(kernel_generator_t));
    if (!gen)
        return NULL;

    gen->final_output = final_output;
    gen->block_size = block_size;
    gen->grid_size = 0;
    gen->analyzed = 0;
    gen->generated = 0;

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

void multi_kernel_destroy(multi_kernel_generator *multi)
{
    if (!multi)
        return;

    for (int i = 0; i < multi->kernel_count; i++)
    {
        if (multi->kernels[i])
        {
            kernel_generator_destroy(multi->kernels[i]);
        }
    }

    if (multi->kernels)
        efree(multi->kernels);

    tensor_list_destroy(&multi->inter_kernel_tensors);

    if (multi->header_code)
        efree(multi->header_code);
    if (multi->device_code)
        efree(multi->device_code);
    if (multi->combined_kernel_code)
        efree(multi->combined_kernel_code);
    if (multi->combined_launch_code)
        efree(multi->combined_launch_code);

    efree(multi);
}


static void kernel_generator_destroy(kernel_generator_t *gen)
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

static bool can_fuse_models(kernel_model_t current, kernel_model_t next)
{
    if (current == next)
        return true;
    if (current == MODEL_ELEMENT_WISE && next == METADATA_TRANSFORM)
        return true;
    if (current == METADATA_TRANSFORM && next == MODEL_ELEMENT_WISE)
        return true;
    if (current == MODEL_ELEMENT_WISE && next == MODEL_REDUCTION)
        return true;
    if (current == MODEL_REDUCTION && next == MODEL_ELEMENT_WISE)
        return true;

    return false;
}

static bool multi_kernel_add_kernel(multi_kernel_generator *multi, kernel_generator_t *kernel)
{
    if (!multi || !kernel)
        return false;

    if (multi->kernel_count >= multi->kernel_capacity)
    {
        int new_capacity = multi->kernel_capacity * 2;
        kernel_generator_t **new_kernels = (kernel_generator_t **)erealloc(
            multi->kernels,
            new_capacity * sizeof(kernel_generator_t *));
        if (!new_kernels)
            return false;

        multi->kernels = new_kernels;
        multi->kernel_capacity = new_capacity;
    }

    multi->kernels[multi->kernel_count] = kernel;
    multi->kernel_count++;

    return true;
}

bool multi_kernel_analyze_and_split(multi_kernel_generator *multi)
{
    if (!multi || !multi->final_output || multi->analyzed)
        return false;

    op_list_t all_ops = {0};
    op_list_init(&all_ops);

    if (multi->final_output->trace.defining_op)
    {
        collect_dependencies_recursive(multi->final_output->trace.defining_op, &all_ops);
    }
    else
    {
        op_list_destroy(&all_ops);
        multi->analyzed = 1;
        return true;
    }

    if (all_ops.count == 0)
    {
        op_list_destroy(&all_ops);
        multi->analyzed = 1;
        return true;
    }

    operation_t *op = NULL;
    kernel_generator_t *current_kernel = NULL;
    kernel_model_t current_model = MODEL_ELEMENT_WISE;
    int kernel_id = 0;

    KERNEL_OPLIST_FOREACH(all_ops.head, op)
    {
        if (!op)
            continue;

        kernel_model_t op_model = op->model;

        bool create_new_kernel = false;

        if (!current_kernel)
        {
            create_new_kernel = true;
        }
        else if (!can_fuse_models(current_model, op_model))
        {
            create_new_kernel = true;
        }

        if (create_new_kernel)
        {
            current_kernel = kernel_generator_create_single(NULL, multi->default_block_size);
            if (!current_kernel)
            {
                printf("here\n");
                op_list_destroy(&all_ops);
                return false;
            }

            current_kernel->id = kernel_id++;
            current_kernel->kernel_type = op_model;

            if (!multi_kernel_add_kernel(multi, current_kernel))
            {
                kernel_generator_destroy(current_kernel);
                op_list_destroy(&all_ops);
                return false;
            }

            current_model = op_model;
        }

        op_list_append(&current_kernel->required_ops, op);
    }

    for (int i = 0; i < multi->kernel_count; i++)
    {
        kernel_generator_t *kg = multi->kernels[i];
        if (!kg)
            continue;

        kg->block_size = multi->default_block_size;

        if (!analyze_single_kernel(kg))
        {
            php_error_docref(NULL, E_WARNING,
                             "Failed to analyze kernel %d", i);
        }
    }

    identify_inter_kernel_connections(multi);
    calculate_kernel_dependencies(multi);

    op_list_destroy(&all_ops);
    multi->analyzed = 1;

    return true;
}

static bool analyze_single_kernel(kernel_generator_t *kg)
{
    if (!kg || kg->analyzed)
        return false;

    tensor_list_t all_tensors = {0};
    collect_all_tensors_from_kernel(kg, &all_tensors);

    for (int i = 0; i < all_tensors.count; i++)
    {
        tensor_t *t = all_tensors.tensors[i];
        if (!t)
            continue;

        bool is_produced = false;
        op_list_node_t *current = kg->required_ops.head;
        while (current)
        {
            if (current->op && current->op->result == t)
            {
                is_produced = true;
                break;
            }
            current = current->next;
        }

        bool is_consumed = false;
        current = kg->required_ops.head;
        while (current)
        {
            operation_t *op = current->op;
            if (op)
            {
                tensor_t *inputs[2] = {NULL};
                int num_inputs = get_op_inputs(op, inputs);
                for (int j = 0; j < num_inputs; j++)
                {
                    if (inputs[j] == t)
                    {
                        is_consumed = true;
                        break;
                    }
                }
            }
            if (is_consumed)
                break;
            current = current->next;
        }

        if (is_produced)
        {
            t->trace.tensor_type = TENSOR_TYPE_OUTPUT;
            add_tensor_if_new(&kg->outputs, t);
        }
        else if (is_consumed)
        {
            t->trace.tensor_type = TENSOR_TYPE_INPUT;
            add_tensor_if_new(&kg->inputs, t);
        }
        else
        {
            t->trace.tensor_type = TENSOR_TYPE_TEMP;
            add_tensor_if_new(&kg->temps, t);
        }
    }

    if (kg->required_ops.tail && kg->required_ops.tail->op)
    {
        kg->final_output = kg->required_ops.tail->op->result;
    }

    if (kg->final_output)
    {
        size_t elements = get_total_elements(kg->final_output);
        kg->total_threads = (int)elements;
        kg->grid_size = (kg->total_threads + kg->block_size - 1) / kg->block_size;
        if (kg->grid_size == 0)
            kg->grid_size = 1;
    }

    kg->num_ops = kg->required_ops.count;
    kg->num_params = kg->inputs.count + kg->outputs.count;

    kg->memory_bytes = 0;
    for (int i = 0; i < kg->inputs.count; i++)
    {
        if (kg->inputs.tensors[i])
        {
            kg->memory_bytes += get_total_elements(kg->inputs.tensors[i]) * sizeof(float);
        }
    }
    for (int i = 0; i < kg->outputs.count; i++)
    {
        if (kg->outputs.tensors[i])
        {
            kg->memory_bytes += get_total_elements(kg->outputs.tensors[i]) * sizeof(float);
        }
    }
    for (int i = 0; i < kg->temps.count; i++)
    {
        if (kg->temps.tensors[i])
        {
            kg->memory_bytes += get_total_elements(kg->temps.tensors[i]) * sizeof(float);
        }
    }

    kg->analyzed = 1;
    tensor_list_destroy(&all_tensors);

    return true;
}

static void collect_all_tensors_from_kernel(kernel_generator_t *kg, tensor_list_t *list)
{
    if (!kg || !list)
        return;

    op_list_node_t *current = kg->required_ops.head;
    while (current)
    {
        operation_t *op = current->op;
        if (op)
        {
            if (op->result)
            {
                add_tensor_if_new(list, op->result);
            }

            tensor_t *inputs[2] = {NULL};
            int num_inputs = get_op_inputs(op, inputs);

            for (int i = 0; i < num_inputs; i++)
            {
                if (inputs[i])
                {
                    add_tensor_if_new(list, inputs[i]);
                }
            }
        }
        current = current->next;
    }
}

static void identify_inter_kernel_connections(multi_kernel_generator *multi)
{
    if (!multi || multi->kernel_count < 2)
        return;

    tensor_list_init(&multi->inter_kernel_tensors);

    for (int i = 0; i < multi->kernel_count; i++)
    {
        for (int j = i + 1; j < multi->kernel_count; j++)
        {
            kernel_generator_t *producer = multi->kernels[i];
            kernel_generator_t *consumer = multi->kernels[j];

            if (!producer || !consumer)
                continue;

            for (int out_idx = 0; out_idx < producer->outputs.count; out_idx++)
            {
                tensor_t *produced = producer->outputs.tensors[out_idx];
                if (!produced)
                    continue;

                for (int in_idx = 0; in_idx < consumer->inputs.count; in_idx++)
                {
                    tensor_t *consumed = consumer->inputs.tensors[in_idx];
                    if (consumed == produced)
                    {
                        produced->trace.tensor_type = TENSOR_TYPE_OUTPUT;
                        add_tensor_if_new(&multi->inter_kernel_tensors, produced);
                        break;
                    }
                }
            }
        }
    }
}

static void calculate_kernel_dependencies(multi_kernel_generator *multi)
{
    if (!multi)
        return;

    for (int i = 0; i < multi->kernel_count; i++)
    {
        kernel_generator_t *consumer = multi->kernels[i];
        if (!consumer)
            continue;

        int dep_count = 0;
        for (int j = 0; j < i; j++)
        {
            kernel_generator_t *producer = multi->kernels[j];
            if (!producer)
                continue;

            bool has_dependency = false;
            for (int k = 0; k < consumer->inputs.count; k++)
            {
                tensor_t *input = consumer->inputs.tensors[k];
                if (!input)
                    continue;

                for (int l = 0; l < producer->outputs.count; l++)
                {
                    if (producer->outputs.tensors[l] == input)
                    {
                        has_dependency = true;
                        break;
                    }
                }
                if (has_dependency)
                    break;
            }

            if (has_dependency)
            {
                dep_count++;
            }
        }

        if (dep_count > 0)
        {
            consumer->dependencies = (kernel_generator_t **)ecalloc(
                dep_count, sizeof(kernel_generator_t *));
            consumer->num_dependencies = dep_count;

            int dep_idx = 0;
            for (int j = 0; j < i; j++)
            {
                kernel_generator_t *producer = multi->kernels[j];
                if (!producer)
                    continue;

                bool has_dependency = false;
                for (int k = 0; k < consumer->inputs.count; k++)
                {
                    tensor_t *input = consumer->inputs.tensors[k];
                    if (!input)
                        continue;

                    for (int l = 0; l < producer->outputs.count; l++)
                    {
                        if (producer->outputs.tensors[l] == input)
                        {
                            has_dependency = true;
                            break;
                        }
                    }
                    if (has_dependency)
                        break;
                }

                if (has_dependency)
                {
                    consumer->dependencies[dep_idx++] = producer;
                }
            }
        }
    }
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

static void generate_single_kernel_code(kernel_generator_t *kg, int kernel_id)
{
    if (!kg || !kg->analyzed)
        return;

    if (kg->kernel_code)
        memset(kg->kernel_code, 0, MAX_CODE_SIZE);

    tensor_list_t *inputs = &kg->inputs;
    tensor_list_t *outputs = &kg->outputs;
    tensor_list_t *temps = &kg->temps;

    append_code(&kg->kernel_code, "__global__ void kernel_%d(\n", kernel_id);
    char *param_list = generate_parameter_list(inputs, outputs);
    append_code(&kg->kernel_code, "%s", param_list);
    append_code(&kg->kernel_code, "\n) {\n");
    efree(param_list);

    append_code(&kg->kernel_code, "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n");

    if (kg->total_threads > 0 && (size_t)kg->total_threads < (size_t)kg->block_size * kg->grid_size)
    {
        append_code(&kg->kernel_code, "  if (idx >= %d) return;\n\n", kg->total_threads);
    }

    if (temps->count > 0)
    {
        append_code(&kg->kernel_code, "  // Temporary registers\n");
        for (int i = 0; i < temps->count; i++)
        {
            tensor_t *t = temps->tensors[i];
            if (t && t->trace.expr_alias[0] != '\0')
            {
                append_code(&kg->kernel_code, "  float %s;\n", t->trace.expr_alias);
            }
        }
        append_code(&kg->kernel_code, "\n");
    }

    append_code(&kg->kernel_code, "  // Operations\n");

    operation_t *op;
    KERNEL_OPLIST_FOREACH(kg->required_ops.head, op)
    {
        char *expr = kernel_define_op(op);
        if (expr)
        {
            append_code(&kg->kernel_code, "%s\n", expr);
            efree(expr);
        }
    }

    append_code(&kg->kernel_code, "}\n");
}

static void generate_single_launch_code(kernel_generator_t *kg, int kernel_id)
{
    if (!kg)
        return;

    if (kg->launch_code)
        memset(kg->launch_code, 0, MAX_CODE_SIZE);

    append_code(&kg->launch_code,
                "// Launch kernel %d\n"
                "{\n"
                "    dim3 blockDim_%d(%d);\n"
                "    dim3 gridDim_%d(%d);\n"
                "    kernel_%d<<<gridDim_%d, blockDim_%d>>>(\n",
                kernel_id, kernel_id, kg->block_size,
                kernel_id, kg->grid_size,
                kernel_id, kernel_id, kernel_id);

    int launch_count = 0;
    for (int i = 0; i < kg->inputs.count; i++)
    {
        tensor_t *t = kg->inputs.tensors[i];
        if (t && t->trace.expr_alias[0] != '\0')
        {
            append_code(&kg->launch_code, "        %s%s", t->trace.expr_alias,
                        (launch_count < kg->num_params - 1) ? ",\n" : "");
            launch_count++;
        }
    }

    for (int i = 0; i < kg->outputs.count; i++)
    {
        tensor_t *t = kg->outputs.tensors[i];
        if (t && t->trace.expr_alias[0] != '\0')
        {
            append_code(&kg->launch_code, "        %s%s", t->trace.expr_alias,
                        (launch_count < kg->num_params - 1) ? ",\n" : "\n");
            launch_count++;
        }
    }

    append_code(&kg->launch_code, "    );\n");
    append_code(&kg->launch_code, "    cudaDeviceSynchronize();\n");
    append_code(&kg->launch_code, "}\n");
}

bool multi_kernel_generate(multi_kernel_generator *multi)
{
    if (!multi || !multi->analyzed || multi->generated)
        return false;

    if (multi->header_code)
        memset(multi->header_code, 0, MAX_CODE_SIZE);
    if (multi->device_code)
        memset(multi->device_code, 0, MAX_CODE_SIZE);
    if (multi->combined_kernel_code)
        memset(multi->combined_kernel_code, 0, MAX_CODE_SIZE);
    if (multi->combined_launch_code)
        memset(multi->combined_launch_code, 0, MAX_CODE_SIZE);

    append_code(&multi->header_code,
                "// Generated CUDA Multi-Kernel\n"
                "#include <cuda_runtime.h>\n"
                "#include <cmath>\n\n");

    append_code(&multi->device_code,
                "// Device functions\n"
                "__device__ float cuda_safe_div(float a, float b) {\n"
                "    return b != 0.0f ? a / b : 0.0f;\n"
                "}\n\n");

    for (int i = 0; i < multi->kernel_count; i++)
    {
        kernel_generator_t *kg = multi->kernels[i];
        if (!kg)
            continue;

        generate_single_kernel_code(kg, i);
        generate_single_launch_code(kg, i);

        append_code(&multi->combined_kernel_code, "%s\n", kg->kernel_code);

        if (i > 0)
            append_code(&multi->combined_launch_code, "\n");
        append_code(&multi->combined_launch_code, "%s", kg->launch_code);

        kg->generated = 1;
    }

    multi->generated = 1;
    return true;
}

kernel_generator_t *kernel_generator_create(tensor_t *final_output)
{
    multi_kernel_generator *multi = multi_kernel_create(final_output);
    if (!multi)
        return NULL;

    if (multi->kernel_count == 0)
    {
        kernel_generator_t *first_kernel = kernel_generator_create_single(final_output, 256);
        if (!first_kernel)
        {
            multi_kernel_destroy(multi);
            return NULL;
        }
        multi_kernel_add_kernel(multi, first_kernel);
    }

    return multi->kernels[0];
}

bool kernel_generator_analyze(kernel_generator_t *gen)
{
    multi_kernel_generator *multi = (multi_kernel_generator *)gen;
    return multi_kernel_analyze_and_split(multi);
}

bool kernel_generator_generate(kernel_generator_t *gen)
{
    multi_kernel_generator *multi = (multi_kernel_generator *)gen;
    return multi_kernel_generate(multi);
}

void multi_kernel_generator_print(multi_kernel_generator *multi)
{
    if (!multi)
        return;

    php_printf("\n=== GENERATED CUDA MULTI-KERNEL ===\n\n");
    php_printf("// Number of kernels: %d\n", multi->kernel_count);

    for (int i = 0; i < multi->kernel_count; i++)
    {
        kernel_generator_t *kg = multi->kernels[i];
        if (!kg)
            continue;

        php_printf("\n// Kernel %d:\n", i);
        php_printf("//   Operations: %d\n", kg->num_ops);
        php_printf("//   Inputs: %d, Outputs: %d, Temps: %d\n",
                   kg->inputs.count, kg->outputs.count, kg->temps.count);
        php_printf("//   Threads: %d (%d blocks x %d threads)\n",
                   kg->total_threads, kg->grid_size, kg->block_size);
        php_printf("//   Memory: ~%zu bytes\n", kg->memory_bytes);
    }

    php_printf("\n// HEADER\n%s\n", multi->header_code);
    php_printf("// DEVICE FUNCTIONS\n%s\n", multi->device_code);
    php_printf("// COMBINED KERNELS\n%s\n", multi->combined_kernel_code);
    php_printf("// COMBINED LAUNCH CONFIG\n%s\n", multi->combined_launch_code);
    php_printf("=======================================\n");
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