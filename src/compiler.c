#include "compiler.h"
#include "compiler_arginfo.h"
#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "zend_closures.h"
#include "kernel_reflection.h"
#include "ast_cuda_compiler.h"
#include "zend_ast.h"
#include "zend_compile.h"
#include "ext/standard/php_smart_string.h"
#include "cuda_globals.h"

#include <nvrtc.h>
#include <cuda.h>

zend_class_entry *cuda_compiler_ce;
extern zend_class_entry *cuda_attr_input_ce;
extern zend_class_entry *cuda_attr_output_ce;
static zend_object_handlers compiler_handlers;

static void compiler_free_object(zend_object *object);
static zend_object *compiler_create_object(zend_class_entry *class_type);

static const char *get_nvrtc_error_string(nvrtcResult result)
{
    switch (result)
    {
    case NVRTC_SUCCESS:
        return "NVRTC_SUCCESS";
    case NVRTC_ERROR_OUT_OF_MEMORY:
        return "NVRTC_ERROR_OUT_OF_MEMORY";
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
        return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case NVRTC_ERROR_INVALID_INPUT:
        return "NVRTC_ERROR_INVALID_INPUT";
    case NVRTC_ERROR_INVALID_PROGRAM:
        return "NVRTC_ERROR_INVALID_PROGRAM";
    case NVRTC_ERROR_INVALID_OPTION:
        return "NVRTC_ERROR_INVALID_OPTION";
    case NVRTC_ERROR_COMPILATION:
        return "NVRTC_ERROR_COMPILATION";
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
        return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
        return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
        return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case NVRTC_ERROR_INTERNAL_ERROR:
        return "NVRTC_ERROR_INTERNAL_ERROR";
    default:
        return "Unknown NVRTC error";
    }
}

static const char *get_cuda_error_string(CUresult result)
{
    switch (result)
    {
    case CUDA_SUCCESS:
        return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
        return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
        return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
        return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
        return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED:
        return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    default:
        return "Unknown CUDA error";
    }
}

static int indent_level = 0;
static void print_indent()
{
    for (int i = 0; i < indent_level; i++)
    {
        php_printf("  ");
    }
}

static const char *get_ast_kind_name(zend_ast *ast)
{
    if (ast == NULL)
        return "NULL";

    switch (ast->kind)
    {
    case ZEND_AST_ZVAL:
        return "LITERAL";
    case ZEND_AST_CONSTANT:
        return "CONSTANT";
    case ZEND_AST_STMT_LIST:
        return "STMT_LIST";
    case ZEND_AST_IF:
        return "IF";
    case ZEND_AST_IF_ELEM:
        return "IF_ELEM";
    case ZEND_AST_WHILE:
        return "WHILE";
    case ZEND_AST_DO_WHILE:
        return "DO_WHILE";
    case ZEND_AST_FOR:
        return "FOR";
    case ZEND_AST_ARG_LIST:
        return "ARG_LIST";
    case ZEND_AST_EXPR_LIST:
        return "EXPR_LIST";

    case ZEND_AST_VAR:
        return "VAR";
    case ZEND_AST_RETURN:
        return "RETURN";
    case ZEND_AST_CONST:
        return "CONST_FETCH";

    case ZEND_AST_DIM:
        return "ARRAY_ACCESS";
    case ZEND_AST_CALL:
        return "CALL";

    case ZEND_AST_ASSIGN:
        return "ASSIGN";
    case ZEND_AST_ASSIGN_OP:
        return "ASSIGN_OP";
    case ZEND_AST_BINARY_OP:
        return "BINARY_OP";

    case ZEND_AST_GREATER:
        return "OP_GREATER";
    case ZEND_AST_GREATER_EQUAL:
        return "OP_GEQUAL";
    case ZEND_AST_AND:
        return "OP_AND";
    case ZEND_AST_OR:
        return "OP_OR";

    case ZEND_AST_METHOD_CALL:
        return "METHOD_CALL";
    case ZEND_AST_CONDITIONAL:
        return "TERNARY_OP";
    case ZEND_AST_POST_DEC:
        return "POST_DEC";
    case ZEND_AST_POST_INC:
        return "POST_INC";

    default:
        php_printf("UNKNOWN_KIND(%d)", ast->kind);
        return "UNKNOWN";
    }
}

static char *read_entire_file(const char *filename, size_t *out_len)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char *)emalloc(file_size + 1);
    size_t read_size = fread(buffer, 1, file_size, file);
    buffer[read_size] = '\0';

    fclose(file);

    if (out_len)
    {
        *out_len = read_size;
    }

    return buffer;
}

static void print_ast_recursive(zend_ast *ast)
{
    if (ast == NULL)
        return;

    print_indent();

    php_printf("[%s] ", get_ast_kind_name(ast));

    if (ast->kind == ZEND_AST_ZVAL)
    {
        zend_ast_zval *zval_node = (zend_ast_zval *)ast;
        php_printf("Value: ");
        zend_print_zval(&zval_node->val, 0);
        php_printf("\n");
        return;
    }

    if (ast->kind == ZEND_AST_VAR)
    {
        php_printf("Variable:\n");
        indent_level++;
        print_ast_recursive(ast->child[0]);
        indent_level--;
        return;
    }

    if (ast->kind == ZEND_AST_BINARY_OP)
    {
        php_printf("Opcode: %d\n", ast->attr);
    }
    else
    {
        php_printf("\n");
    }

    indent_level++;
    if (zend_ast_is_list(ast))
    {
        zend_ast_list *list = (zend_ast_list *)ast;
        for (uint32_t i = 0; i < list->children; i++)
        {
            print_ast_recursive(list->child[i]);
        }
    }
    else
    {
        uint32_t children = zend_ast_get_num_children(ast);
        for (uint32_t i = 0; i < children; i++)
        {
            print_ast_recursive(ast->child[i]);
        }
    }

    indent_level--;
}

static char *extract_function_body_for_ast(
    const char *source,
    size_t source_len,
    uint32_t start_line,
    uint32_t end_line,
    size_t *out_len)
{
    if (start_line == 0 || end_line == 0 || start_line > end_line)
    {
        return NULL;
    }

    const char **line_offsets = emalloc(sizeof(char *) * (end_line + 3));
    uint32_t current_line = 1;
    line_offsets[1] = source;

    for (size_t i = 0; i < source_len; i++)
    {
        if (source[i] == '\n')
        {
            current_line++;
            if (current_line > end_line + 1)
                break;
            line_offsets[current_line] = &source[i + 1];
        }
    }

    if (end_line > current_line)
    {
        efree(line_offsets);
        return NULL;
    }

    const char *func_start = line_offsets[start_line];
    const char *func_end =
        (end_line < current_line) ? line_offsets[end_line + 1] : source + source_len;

    const char *body_start = NULL;
    const char *body_end = NULL;
    int brace_level = 0;
    int found_open = 0;

    for (const char *p = func_start; p < func_end; p++)
    {
        if (*p == '{')
        {
            if (!found_open)
            {
                body_start = p + 1;
                found_open = 1;
            }
            brace_level++;
        }
        else if (*p == '}')
        {
            brace_level--;
            if (brace_level == 0 && found_open)
            {
                body_end = p;
                break;
            }
        }
    }

    if (!body_start || !body_end || body_end <= body_start)
    {
        efree(line_offsets);
        return NULL;
    }

    size_t body_len = body_end - body_start;

    const char *prefix = "<?php\n";
    size_t prefix_len = strlen(prefix);

    char *out = emalloc(prefix_len + body_len + 1);

    memcpy(out, prefix, prefix_len);
    memcpy(out + prefix_len, body_start, body_len);

    out[prefix_len + body_len] = '\0';

    if (out_len)
    {
        *out_len = prefix_len + body_len;
    }

    efree(line_offsets);
    return out;
}

static char *build_complete_cuda_program(cuda_compiler_object *compiler, size_t *out_len)
{
    smart_string program = {0};

    smart_string_appendl(&program, "#include <cuda_runtime.h>\n", strlen("#include <cuda_runtime.h>\n"));
    smart_string_appendl(&program, "#include <device_launch_parameters.h>\n", strlen("#include <device_launch_parameters.h>\n"));
    smart_string_appendl(&program, "#include <cuda_fp16.h>\n", strlen("#include <cuda_fp16.h>\n"));
    smart_string_appendl(&program, "\n// Math function wrappers\n", strlen("\n// Math function wrappers\n"));

    smart_string_appendl(&program, "// Math constants for CUDA\n", strlen("// Math constants for CUDA\n"));
    smart_string_appendl(&program, "#ifndef M_PI\n", strlen("#ifndef M_PI\n"));
    smart_string_appendl(&program, "#define M_PI 3.14159265358979323846f\n", strlen("#define M_PI 3.14159265358979323846f\n"));
    smart_string_appendl(&program, "#endif\n\n", strlen("#endif\n\n"));

    smart_string_appendl(&program, "#ifndef INFINITY\n", strlen("#ifndef INFINITY\n"));
    smart_string_appendl(&program, "#define INFINITY __int_as_float(0x7f800000)\n", strlen("#define INFINITY __int_as_float(0x7f800000)\n"));
    smart_string_appendl(&program, "#endif\n\n", strlen("#endif\n\n"));

    smart_string_appendl(&program, "#ifndef NAN\n", strlen("#ifndef NAN\n"));
    smart_string_appendl(&program, "#define NAN __int_as_float(0x7fffffff)\n", strlen("#define NAN __int_as_float(0x7fffffff)\n"));
    smart_string_appendl(&program, "#endif\n\n", strlen("#endif\n\n"));

    smart_string_appendl(&program, "#ifndef FLT_MAX\n", strlen("#ifndef FLT_MAX\n"));
    smart_string_appendl(&program, "#define FLT_MAX 3.402823466e+38f\n", strlen("#define FLT_MAX 3.402823466e+38f\n"));
    smart_string_appendl(&program, "#endif\n\n", strlen("#endif\n\n"));

    smart_string_appendl(&program, "#ifndef FLT_MIN\n", strlen("#ifndef FLT_MIN\n"));
    smart_string_appendl(&program, "#define FLT_MIN 1.175494351e-38f\n", strlen("#define FLT_MIN 1.175494351e-38f\n"));
    smart_string_appendl(&program, "#endif\n\n", strlen("#endif\n\n"));

    smart_string_appendl(&program, "#ifndef INF\n", strlen("#ifndef INF\n"));
    smart_string_appendl(&program, "#define INF INFINITY\n", strlen("#define INF INFINITY\n"));
    smart_string_appendl(&program, "#endif\n\n", strlen("#endif\n\n"));

    zval *header_zv;
    ZEND_HASH_FOREACH_VAL(compiler->headers, header_zv)
    {
        if (Z_TYPE_P(header_zv) == IS_STRING)
        {
            smart_string_appendl(&program, Z_STRVAL_P(header_zv), Z_STRLEN_P(header_zv));
            smart_string_appendc(&program, '\n');
        }
    }
    ZEND_HASH_FOREACH_END();

    cuda_device_object *device;
    ZEND_HASH_FOREACH_PTR(compiler->devices, device)
    {
        smart_string_appendl(&program, "\n// Device function: ", strlen("\n// Device function: "));
        smart_string_appendl(&program, ZSTR_VAL(device->name), ZSTR_LEN(device->name));
        smart_string_appendl(&program, "\n", 1);
    }
    ZEND_HASH_FOREACH_END();

    cuda_kernel_data *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        smart_string_appendl(&program, "\n// Kernel: ", strlen("\n// Kernel: "));
        smart_string_appendl(&program, ZSTR_VAL(kernel->name), ZSTR_LEN(kernel->name));
        smart_string_appendl(&program, "\n", 1);

        if (kernel->cuda_code)
        {
            smart_string_appendl(&program, kernel->cuda_code, strlen(kernel->cuda_code));
        }
        smart_string_appendl(&program, "\n", 1);
    }
    ZEND_HASH_FOREACH_END();

    smart_string_0(&program);

    if (out_len)
    {
        *out_len = program.len;
    }

    return program.c;
}

ZEND_METHOD(Compiler, __construct)
{
    cuda_compiler_object *compiler;
    zend_string *target_str = NULL;
    zend_long optimization = 3;
    zend_bool debug = 0;
    zend_bool fast_math = 1;

    ZEND_PARSE_PARAMETERS_START(0, 4)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR_OR_NULL(target_str)
    Z_PARAM_LONG(optimization)
    Z_PARAM_BOOL(debug)
    Z_PARAM_BOOL(fast_math)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    compiler->kernels = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->kernels, 8, NULL, NULL, 0);

    compiler->devices = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->devices, 8, NULL, NULL, 0);

    if (target_str)
    {
        compiler->target_device = estrndup(ZSTR_VAL(target_str), ZSTR_LEN(target_str));
    }
    else
    {
        compiler->target_device = estrdup("sm_60");
    }

    compiler->optimization_level = optimization;
    compiler->debug_mode = debug;
    compiler->fast_math = fast_math;
    compiler->compilation_context = NULL;
}

ZEND_METHOD(Compiler, kernel)
{
    cuda_compiler_object *compiler;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_string *name = NULL;
    zend_string *target = NULL;
    zval *grid_zv = NULL;
    zval *block_zv = NULL;

    zend_function *fptr = fcc.function_handler;
    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        return;
    }

    if (fptr->op_array.fn_flags & ZEND_ACC_USES_THIS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel functions cannot use object context.");
        return;
    }

    if (fptr->op_array.static_variables != NULL)
    {
        zend_throw_exception_ex(
            NULL,
            0,
            "Cuda Runtime cannot outter context variables.");
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_kernel_ce);
    func_parameter_list_t *params = cuda_extract_parameters(fptr);

    cuda_compilation_context_t *ctx = create_cuda_context(params, FN_KERNEL, fargs->name, compiler->headers);

    if (!fargs)
    {
        return;
    }

    zend_op_array *op_array = &fptr->op_array;
    if (!op_array->filename || op_array->line_start == 0 || op_array->line_end == 0)
    {
        zend_string_release(name);
        zend_string_release(target);
        zend_string_release(fargs->name);
        zend_string_release(fargs->target);
        efree(fargs);
        return;
    }

    size_t file_len = 0;
    char *file_content = read_entire_file(ZSTR_VAL(op_array->filename), &file_len);
    if (!file_content)
    {
        zend_string_release(name);
        zend_string_release(target);
        zend_string_release(fargs->name);
        zend_string_release(fargs->target);
        efree(fargs);
        return;
    }

    size_t src_len = 0;
    char *src = extract_function_body_for_ast(
        file_content, file_len,
        op_array->line_start, op_array->line_end,
        &src_len);

    efree(file_content);

    zend_string *source_code = NULL;
    if (src)
    {
        source_code = zend_string_init(src, src_len, 0);
        efree(src);
    }
    else
    {
        const char *fallback = "<?php\nreturn 0;";
        source_code = zend_string_init(fallback, strlen(fallback), 0);
    }

    zend_arena *ast_arena = NULL;
    zend_ast *ast = zend_compile_string_to_ast(source_code, &ast_arena, fargs->name);

    if (compile_ast_to_cuda_fn(ctx, ast) != 1)
    {
        return;
    }

    smart_string_0(ctx->cuda_code_buffer);

    cuda_kernel_data *kernel = (cuda_kernel_data *)ecalloc(1, sizeof(cuda_kernel_data));
    kernel->name = zend_string_copy(fargs->name);
    kernel->target = zend_string_copy(fargs->target);
    kernel->fci = fci;
    kernel->fcc = fcc;

    kernel->grid[0] = 4;
    kernel->grid[1] = 4;
    kernel->grid[2] = 4;
    kernel->block[0] = 16;
    kernel->block[1] = 16;
    kernel->block[2] = 16;

    if (grid_zv && Z_TYPE_P(grid_zv) == IS_ARRAY)
    {
        zval *x = zend_hash_index_find(Z_ARR_P(grid_zv), 0);
        zval *y = zend_hash_index_find(Z_ARR_P(grid_zv), 1);
        zval *z = zend_hash_index_find(Z_ARR_P(grid_zv), 2);
        if (x)
            kernel->grid[0] = zval_get_long(x);
        if (y)
            kernel->grid[1] = zval_get_long(y);
        if (z)
            kernel->grid[2] = zval_get_long(z);
    }

    if (block_zv && Z_TYPE_P(block_zv) == IS_ARRAY)
    {
        zval *x = zend_hash_index_find(Z_ARR_P(block_zv), 0);
        zval *y = zend_hash_index_find(Z_ARR_P(block_zv), 1);
        zval *z = zend_hash_index_find(Z_ARR_P(block_zv), 2);
        if (x)
            kernel->block[0] = zval_get_long(x);
        if (y)
            kernel->block[1] = zval_get_long(y);
        if (z)
            kernel->block[2] = zval_get_long(z);
    }

    kernel->ast = ast;
    kernel->ast_arena = ast_arena;
    kernel->source_code = source_code;
    kernel->cuda_code = ctx->cuda_code_buffer->c;

    kernel->parameters = cuda_extract_parameter(fptr);

    kernel->used_devices = emalloc(sizeof(HashTable));
    zend_hash_init(kernel->used_devices, 4, NULL, NULL, 0);

    zend_hash_str_add_ptr(compiler->kernels,
                          ZSTR_VAL(kernel->name),
                          ZSTR_LEN(kernel->name),
                          kernel);

    zend_string_release(fargs->name);
    zend_string_release(fargs->target);
    efree(fargs);

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, device)
{
    cuda_compiler_object *compiler;
    zend_string *name;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zval *attributes = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_function *fptr = fcc.function_handler;
    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
    if (!fargs)
    {
        return;
    }

    zend_string *device_class_name = zend_string_init("Cuda\\Device", strlen("Cuda\\Device"), 0);
    zend_class_entry *device_ce = zend_lookup_class(device_class_name);
    zend_string_release(device_class_name);

    if (!device_ce)
    {
        zend_throw_exception_ex(NULL, 0, "Class Cuda\\Device not found");
        RETURN_NULL();
    }

    zval device_zv;
    object_init_ex(&device_zv, device_ce);

    cuda_device_object *device = Z_CUDA_DEVICE_P(&device_zv);

    device->fci = fci;
    device->fcc = fcc;
    device->name = zend_string_copy(fargs->name);
    device->target = zend_string_copy(fargs->target);
    device->ast = NULL;
    device->ast_arena = NULL;

    zend_hash_add_ptr(compiler->devices, zend_string_copy(fargs->name), device);
    zend_string_release(fargs->name);
    zend_string_release(fargs->target);
    efree(fargs);

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, compile)
{
    cuda_compiler_object *compiler;
    char *target = NULL;
    size_t target_len;
    zend_bool optimize = 1;
    zend_bool debug = 0;

    ZEND_PARSE_PARAMETERS_START(0, 3)
    Z_PARAM_OPTIONAL
    Z_PARAM_STRING(target, target_len)
    Z_PARAM_BOOL(optimize)
    Z_PARAM_BOOL(debug)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    size_t program_len;
    char *cuda_program = build_complete_cuda_program(compiler, &program_len);

    if (!cuda_program || program_len == 0)
    {
        if (cuda_program)
            efree(cuda_program);
        zend_throw_exception_ex(NULL, 0, "Failed to build CUDA program");
        RETURN_NULL();
    }

    smart_string opts = {0};

    if (target && target_len > 0)
    {
        smart_string_appendl(&opts, "-arch=", 6);
        smart_string_appendl(&opts, target, target_len);
        smart_string_appendc(&opts, ' ');
    }
    else if (compiler->target_device)
    {
        smart_string_appendl(&opts, "-arch=", 6);
        smart_string_appendl(&opts, compiler->target_device, strlen(compiler->target_device));
        smart_string_appendc(&opts, ' ');
    }
    else
    {
        smart_string_appendl(&opts, "-arch=sm_60 ", 12);
    }

    if (optimize && !debug)
    {
        if (compiler->fast_math)
        {
            smart_string_appendl(&opts, "-use_fast_math ", 15);
        }
    }
    else if (debug)
    {
        smart_string_appendl(&opts, "-G ", 3);
        smart_string_appendl(&opts, "-lineinfo ", 10);
    }
    else
    {
        char opt_level[8];
        snprintf(opt_level, sizeof(opt_level), "-O%d ", compiler->optimization_level);
        smart_string_appendl(&opts, opt_level, strlen(opt_level));
    }

    char include_path[512];

    smart_string_appendl(&opts, CUDA_INCLUDE_PATH_STR, strlen(CUDA_INCLUDE_PATH_STR));
    smart_string_appendc(&opts, ' ');

    smart_string_appendl(&opts, CUDA_CRT_INCLUDE_STR, strlen(CUDA_CRT_INCLUDE_STR));
    smart_string_appendc(&opts, ' ');
    smart_string_appendl(&opts, "-I. ", 4);
    smart_string_appendl(&opts, "-std=c++11 ", 11);
    smart_string_appendl(&opts, "-restrict ", 10);

    smart_string_0(&opts);

    const int MAX_OPTIONS = 32;
    const char *options[MAX_OPTIONS];
    int option_count = 0;

    char *token = strtok(opts.c, " ");
    while (token && option_count < MAX_OPTIONS)
    {
        options[option_count++] = token;
        token = strtok(NULL, " ");
    }

    if (debug)
    {
        php_printf("NVRTC Compilation Options (%d):\n", option_count);
        for (int i = 0; i < option_count; i++)
        {
            php_printf("  [%d] %s\n", i, options[i]);
        }
        php_printf("\nCUDA Program (first 1000 chars):\n%.1000s\n", cuda_program);
    }

    nvrtcProgram prog;
    nvrtcResult nvrtc_result;

    nvrtc_result = nvrtcCreateProgram(&prog,
                                      cuda_program,
                                      "kernel.cu",
                                      0,
                                      NULL,
                                      NULL);

    if (nvrtc_result != NVRTC_SUCCESS)
    {
        efree(cuda_program);
        smart_string_free(&opts);
        zend_throw_exception_ex(NULL, 0,
                                "NVRTC program creation failed: %s (code: %d)",
                                get_nvrtc_error_string(nvrtc_result), nvrtc_result);
        RETURN_NULL();
    }

    nvrtc_result = nvrtcCompileProgram(prog, option_count, options);

    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);

    if (log_size > 1)
    {
        char *compile_log = (char *)emalloc(log_size);
        nvrtcGetProgramLog(prog, compile_log);

        if (debug || strstr(compile_log, "error") || strstr(compile_log, "Error"))
        {
            php_printf("NVRTC Compilation Log:\n%s\n", compile_log);
        }

        efree(compile_log);
    }

    if (nvrtc_result != NVRTC_SUCCESS)
    {
        smart_string_free(&opts);
        efree(cuda_program);
        nvrtcDestroyProgram(&prog);
        zend_throw_exception_ex(NULL, 0,
                                "NVRTC compilation failed: %s (code: %d)",
                                get_nvrtc_error_string(nvrtc_result), nvrtc_result);
        RETURN_NULL();
    }

    size_t ptx_size;
    nvrtc_result = nvrtcGetPTXSize(prog, &ptx_size);

    if (nvrtc_result != NVRTC_SUCCESS)
    {
        smart_string_free(&opts);
        efree(cuda_program);
        nvrtcDestroyProgram(&prog);
        zend_throw_exception_ex(NULL, 0,
                                "Failed to get PTX size: %s",
                                get_nvrtc_error_string(nvrtc_result));
        RETURN_NULL();
    }

    char *ptx_code = (char *)emalloc(ptx_size + 1);
    nvrtc_result = nvrtcGetPTX(prog, ptx_code);
    ptx_code[ptx_size] = '\0';

    smart_string_free(&opts);
    efree(cuda_program);
    nvrtcDestroyProgram(&prog);

    if (nvrtc_result != NVRTC_SUCCESS)
    {
        efree(ptx_code);
        zend_throw_exception_ex(NULL, 0,
                                "Failed to get PTX code: %s",
                                get_nvrtc_error_string(nvrtc_result));
        RETURN_NULL();
    }

    zend_string *module_class_name = zend_string_init("Cuda\\CompiledModule",
                                                      strlen("Cuda\\CompiledModule"), 0);
    zend_class_entry *module_ce = zend_lookup_class(module_class_name);
    zend_string_release(module_class_name);

    if (!module_ce)
    {
        efree(ptx_code);
        zend_throw_exception_ex(NULL, 0,
                                "CompiledModule class not found");
        return;
    }

    zval module_zv;
    object_init_ex(&module_zv, module_ce);
    cuda_module_object *module = Z_CUDA_MODULE_P(&module_zv);

    module->ptx_code = ptx_code;
    module->ptx_size = ptx_size;

    module->kernel_functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

    module->kernel_functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

    cuda_kernel_data *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        if (!kernel || !kernel->name)
            continue;

        cuda_kernel_data *kernel_copy = (cuda_kernel_data *)emalloc(sizeof(cuda_kernel_data));
        memset(kernel_copy, 0, sizeof(cuda_kernel_data));

        kernel_copy->name = zend_string_copy(kernel->name);
        if (kernel->target)
        {
            kernel_copy->target = zend_string_copy(kernel->target);
        }

        kernel_copy->fcc = kernel->fcc;
        kernel_copy->fcc = kernel->fcc;

        memcpy(kernel_copy->grid, kernel->grid, sizeof(kernel->grid));
        memcpy(kernel_copy->block, kernel->block, sizeof(kernel->block));
        kernel_copy->source_code = kernel->source_code ? zend_string_copy(kernel->source_code) : NULL;

        if (kernel->cuda_code)
        {
            kernel_copy->cuda_code = estrdup(kernel->cuda_code);
        }

        kernel_copy->parameters = kernel->parameters;
        kernel_copy->used_devices = kernel->used_devices;
        zend_hash_add_ptr(module->kernel_functions, kernel->name, kernel_copy);
    }
    ZEND_HASH_FOREACH_END();

    if (debug)
    {
        CUresult cu_result = cuInit(0);
        if (cu_result == CUDA_SUCCESS)
        {
            CUdevice cuDevice;
            cu_result = cuDeviceGet(&cuDevice, 0);

            if (cu_result == CUDA_SUCCESS)
            {
                CUcontext cuContext;
                cu_result = cuCtxCreate(&cuContext, 0, cuDevice);

                if (cu_result == CUDA_SUCCESS)
                {
                    CUmodule cuModule;
                    cu_result = cuModuleLoadDataEx(&cuModule, ptx_code, 0, NULL, NULL);

                    if (cu_result == CUDA_SUCCESS)
                    {
                        ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
                        {
                            CUfunction cuFunc;
                            cu_result = cuModuleGetFunction(&cuFunc, cuModule,
                                                            ZSTR_VAL(kernel->name));

                            if (cu_result != CUDA_SUCCESS)
                            {
                                php_printf("Warning: Kernel '%s' not found in compiled module\n",
                                           ZSTR_VAL(kernel->name));
                            }
                        }
                        ZEND_HASH_FOREACH_END();

                        cuModuleUnload(cuModule);
                    }
                    cuCtxDestroy(cuContext);
                }
            }
        }
    }

    RETURN_ZVAL(&module_zv, 1, 0);
}

ZEND_METHOD(Compiler, getKernels)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    array_init(return_value);

    cuda_kernel_data *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        zval kernel_info;
        array_init(&kernel_info);

        add_assoc_str(&kernel_info, "name", zend_string_copy(kernel->name));
        add_assoc_str(&kernel_info, "target", zend_string_copy(kernel->target));
        add_assoc_stringl(&kernel_info, "cuda_code", kernel->cuda_code, strlen(kernel->cuda_code));

        zval grid_zv;
        array_init(&grid_zv);
        add_next_index_long(&grid_zv, kernel->grid[0]);
        add_next_index_long(&grid_zv, kernel->grid[1]);
        add_next_index_long(&grid_zv, kernel->grid[2]);
        add_assoc_zval(&kernel_info, "grid", &grid_zv);

        zval block_zv;
        array_init(&block_zv);
        add_next_index_long(&block_zv, kernel->block[0]);
        add_next_index_long(&block_zv, kernel->block[1]);
        add_next_index_long(&block_zv, kernel->block[2]);
        add_assoc_zval(&kernel_info, "block", &block_zv);

        add_assoc_zval(return_value, ZSTR_VAL(kernel->name), &kernel_info);
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(Compiler, getDevices)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    array_init(return_value);

    cuda_device_object *device;
    ZEND_HASH_FOREACH_PTR(compiler->devices, device)
    {
        zval device_zv;
        ZVAL_OBJ(&device_zv, &device->std);
        zend_hash_add(return_value->value.arr, device->name, &device_zv);
        Z_ADDREF(device_zv);
    }
    ZEND_HASH_FOREACH_END();
}

static void compiler_free_object(zend_object *object)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_FROM_OBJ(object);

    if (compiler->target_device)
    {
        efree(compiler->target_device);
    }

    if (compiler->kernels)
    {
        cuda_kernel_object *kernel;
        ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
        {
            // @todo destroy?
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->kernels);
        efree(compiler->kernels);
    }

    if (compiler->devices)
    {
        cuda_device_object *device;
        ZEND_HASH_FOREACH_PTR(compiler->devices, device)
        {
            // @todo destroy?
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->devices);
        efree(compiler->devices);
    }

    if (compiler->compilation_context)
    {
        free_cuda_context(compiler->compilation_context);
    }

    zend_object_std_dtor(&compiler->std);
}

static zend_object *compiler_create_object(zend_class_entry *class_type)
{
    cuda_compiler_object *compiler =
        (cuda_compiler_object *)ecalloc(1, sizeof(cuda_compiler_object));

    zend_object_std_init(&compiler->std, class_type);
    compiler->std.handlers = &compiler_handlers;
    compiler->headers = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(compiler->headers, 8, NULL, NULL, 0);

    compiler->target_device = NULL;
    compiler->optimization_level = 3;
    compiler->debug_mode = 0;
    compiler->fast_math = 1;
    compiler->kernels = NULL;
    compiler->devices = NULL;
    compiler->compilation_context = NULL;

    return &compiler->std;
}

int compiler_init()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, COMPILER_CLASS_NAME, compiler_methods);
    cuda_compiler_ce = zend_register_internal_class(&ce);
    cuda_compiler_ce->create_object = compiler_create_object;
    cuda_compiler_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&compiler_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    compiler_handlers.offset = XtOffsetOf(cuda_compiler_object, std);
    compiler_handlers.free_obj = compiler_free_object;

    return 1;
}